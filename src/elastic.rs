use crate::structadd::AddGene;
use crate::structadd::Predict;
use smartcore::linear::elastic_net::ElasticNet;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn elasticnetrun(
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

    let elastictrain =
        ElasticNet::fit(&traindatasets.0, &traindatasets.2, Default::default()).unwrap();

    let elastictrain_test = elastictrain.predict(&traindatasets.1).unwrap();

    let accuracy_elasticnew = accuracy(&traindatasets.3, &elastictrain_test);

    println!("The model accuracy is {}", accuracy_elasticnew);

    let elasticnet = elastictrain.predict(&predictionmatrix).unwrap();

    let mut filewrite = File::create("elastic-netpredict").expect("file not present");

    writeln!(filewrite, "The elasticnet prediction values are as follows")
        .expect("file not present");

    for i in elasticnet.iter() {
        writeln!(filewrite, "{}", i).expect("file not present");
    }

    Ok("The elastic net model has been trained".to_string())
}
