use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "cutseq",
    version = "1.0",
    about = "Hash Regression based identification of antimicrobial
       ************************************************
       Author Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate kmer plot
    Kmerplot {
        /// path to the file
        filepath: String,
        /// kmer for the hashes
        kmer: String,
        /// thread for the analysis
        thread: String,
    },
    /// Logistic Antimicrobial
    MicrobialLog {
        /// path to the file
        pathfilestring: String,
        /// kmers for the anti-microbial
        kmerselect: String,
        /// threshold for the anti-microbial
        thresholdselect: String,
        /// path to the anti-microbial sensitivity
        antimicrobialsensitivity: String,
        /// threads for the analysis
        threads: String,
        /// path to the fastafile for the prediction
        prediction: String,
    },
    /// Elastic Net
    ElasticNet {
        /// fastafile for the antimicrobial
        fasta: String,
        /// sensitivity for the threshold
        sensitivitythreshold: String,
        /// prediction file
        predictionfile: String,
        /// kmer selected
        kmersel: String,
        /// threads for the analysis
        threads: String,
        /// sensitivity assay file
        sensitivityassay: String,
    },
    /// Random Forest for classification
    RandomForest {
        /// fastafile for the antimicrobial
        fasta: String,
        /// sensitivity for the threshold
        sensitivitythreshold: String,
        /// prediction file
        predictionfile: String,
        /// kmer selected
        kmersel: String,
        /// threads for the analysis
        threads: String,
        /// sensitivity assay file
        sensitivityassay: String,
    },
    /// KNN Classifier for classification
    KNNClassifier {
        /// fastafile for the antimicrobial
        fasta: String,
        /// sensitivity for the threshold
        sensitivitythreshold: String,
        /// prediction file
        predictionfile: String,
        /// kmer selected
        kmersel: String,
        /// threads for the analysis
        threads: String,
        /// sensitivity assay file
        sensitivityassay: String,
    },
    /// Candle regression using transformers
    Regression {
        /// fastafile for the antimicrobial
        fasta: String,
        /// sensitivity for the threshold
        sensitivitythreshold: String,
        /// prediction file
        predictionfile: String,
        /// kmer selected
        kmersel: String,
        /// threads for the analysis
        threads: String,
        /// sensitivity assay file
        sensitivityassay: String,
    },
}
