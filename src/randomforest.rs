use crate::AddGene;
use crate::Predict;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn randomforest_run(
    pathfile: &str,
    predictfile: &str,
    kmers: &str,
    sensivitythreshold: &str,
    sensitivityfile: &str,
) -> Result<String, Box<dyn Error>> {
    let fastafile = AddGene {
        fastafile: pathfile.to_string(),
    };
    let predictionfile = Predict {
        predictionfile: predictfile.to_string(),
    };

    let fastamatrix = fastafile
        .generatemers(kmers, sensivitythreshold, sensitivityfile)
        .unwrap();

    let predictionmatrix = predictionfile
        .generatemers(kmers, sensivitythreshold)
        .unwrap();

    let traindatasets = train_test_split(&fastamatrix.0, &fastamatrix.1, 0.2, true, Some(2811));

    let randomforest_train =
        RandomForestClassifier::fit(&traindatasets.0, &traindatasets.2, Default::default())
            .unwrap();
    let randomforest_train_predict = randomforest_train.predict(&traindatasets.1).unwrap();
    let accuracy_model = accuracy(&traindatasets.3, &randomforest_train_predict);

    println!("The accuracy of the model is {}", accuracy_model);

    let actualpredict = randomforest_train.predict(&predictionmatrix).unwrap();

    let mut filewrite = File::create("random-forest").expect("file not present");

    writeln!(
        filewrite,
        "The random forest classifier values are as follows"
    )
    .expect("value not found");

    for i in actualpredict.iter() {
        writeln!(filewrite, "actualpredict {}\n", i).expect("file not present");
    }

    Ok("The random forest has been completed".to_string())
}
