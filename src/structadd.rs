/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct AddGene {
    pub fastafile: String,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Fasta {
    pub header: String,
    pub sequence: String,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Predict {
    pub predictionfile: String,
}
