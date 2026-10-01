mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
use figlet_rs::FIGfont;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
mod compare;
mod plot;
mod structadd;
use crate::plot::kmerplot;
mod sensitivity;
use self::elastic::elasticnetrun;
use self::structadd::AddGene;
use self::structadd::Predict;
mod convert;
mod predict;
use std::fs::File;
use std::io::Write;
mod elastic;
mod knnres;
use crate::knnres::knnrun;
mod randomforest;
use randomforest::randomforest_run;
use smartcore::model_selection::train_test_split;
mod torch;
use crate::torch::burn_run;

/*
Gaurav Sablok
gsablok@proton.me
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("cutseq");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::Kmerplot {
            filepath,
            kmer,
            thread,
        } => {
            let n_threads = thread.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let kmerplot = kmerplot(filepath, kmer).unwrap();
                println!("The qsar modelling has finished: {}", kmerplot);
            });
        }
        Commands::MicrobialLog {
            pathfilestring,
            kmerselect,
            thresholdselect,
            antimicrobialsensitivity,
            threads,
            prediction,
        } => {
            let n_threads = threads.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let pathfile = AddGene {
                    fastafile: pathfilestring.to_string(),
                };

                let logistic_unwrap = pathfile
                    .generatemers(kmerselect, thresholdselect, antimicrobialsensitivity)
                    .unwrap();

                let datasets = train_test_split(
                    &logistic_unwrap.0,
                    &logistic_unwrap.1,
                    0.2,
                    true,
                    Some(2811),
                );

                let logisticwrap =
                    LogisticRegression::fit(&datasets.0, &datasets.2, Default::default()).unwrap();

                let logistictest = logisticwrap.predict(&datasets.1).unwrap();

                let accuracy_logistic = accuracy(&datasets.3, &logistictest);

                println!(
                    "The value for the accuracy of the model is {}",
                    accuracy_logistic
                );

                let logisticpredict = Predict {
                    predictionfile: prediction.to_string(),
                };

                let prediction_matrix = logisticpredict
                    .generatemers(kmerselect, thresholdselect)
                    .unwrap();

                let logreg = logisticwrap.predict(&prediction_matrix).unwrap();
                let mut writefile = File::create("predictionvalues").expect("path not found");
                writeln!(
                    writefile,
                    "The prediction values of the crate are as follow"
                )
                .expect("The file to write was not found");
                for i in logreg.iter() {
                    writeln!(writefile, "value: {}\t", i).expect("The file to write was not found");
                }
            });
        }
        Commands::ElasticNet {
            fasta,
            sensitivitythreshold,
            predictionfile,
            kmersel,
            threads,
            sensitivityassay,
        } => {
            let n_threads = threads.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let elasticrun = elasticnetrun(
                    fasta,
                    predictionfile,
                    kmersel,
                    sensitivitythreshold,
                    sensitivityassay,
                )
                .unwrap();
                println!("The elasticnet run has finished: {}", elasticrun);
            });
        }
        Commands::RandomForest {
            fasta,
            sensitivitythreshold,
            predictionfile,
            kmersel,
            threads,
            sensitivityassay,
        } => {
            let n_threads = threads.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let elasticrun = randomforest_run(
                    fasta,
                    predictionfile,
                    kmersel,
                    sensitivitythreshold,
                    sensitivityassay,
                )
                .unwrap();
                println!("The randomforest  run has finished: {}", elasticrun);
            });
        }
        Commands::KNNClassifier {
            fasta,
            sensitivitythreshold,
            predictionfile,
            kmersel,
            threads,
            sensitivityassay,
        } => {
            let n_threads = threads.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let elasticrun = knnrun(
                    fasta,
                    predictionfile,
                    kmersel,
                    sensitivitythreshold,
                    sensitivityassay,
                )
                .unwrap();
                println!("The knn run has finished: {}", elasticrun);
            });
        }
        Commands::Regression {
            fasta,
            sensitivitythreshold,
            predictionfile,
            kmersel,
            threads,
            sensitivityassay,
        } => {
            let n_threads = threads.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let torchrun = burn_run(
                    fasta,
                    predictionfile,
                    kmersel,
                    sensitivitythreshold,
                    sensitivityassay,
                )
                .unwrap();
                println!("The knn run has finished: {}", torchrun);
            });
        }
    }
}
