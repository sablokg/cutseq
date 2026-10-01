use crate::convert::convertarray;
use crate::sensitivity::sensitivity;
use crate::structadd::Fasta;
use crate::structadd::Predict;
use core::result::Result;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

impl Predict {
    pub fn make_matrix(&self) -> Result<Vec<Fasta>, Box<dyn Error>> {
        let pathfile = File::open(self.predictionfile.to_string()).expect("fasta file not found");
        let pathfileread = BufReader::new(pathfile);

        let mut pathvec: Vec<Fasta> = Vec::new();
        let mut header: Vec<String> = Vec::new();
        let mut sequence: Vec<String> = Vec::new();

        for i in pathfileread.lines() {
            let line = i.expect("line not found");
            if line.starts_with(">") {
                let lineheader = line.split(">").collect::<Vec<_>>()[0].to_string();
                header.push(lineheader);
            }
            if !line.starts_with(">") {
                sequence.push(line);
            }
        }

        for i in 0..header.len() {
            pathvec.push(Fasta {
                header: header[i].clone(),
                sequence: sequence[i].clone(),
            })
        }
        Ok(pathvec)
    }

    pub fn generatemers(
        &self,
        kmer: &str,
        sensivity: &str,
    ) -> Result<DenseMatrix<f64>, Box<dyn Error>> {
        let vecunpack = self.make_matrix().unwrap();

        let mut newhashset: HashSet<Vec<_>> = HashSet::new();

        for i in vecunpack.iter() {
            let seqval = i
                .sequence
                .as_bytes()
                .windows(kmer.parse::<usize>().unwrap())
                .map(|x| str::from_utf8(x).unwrap())
                .collect::<Vec<_>>();
            newhashset.insert(seqval.clone());
        }

        let mut finalhashset: HashSet<String> = HashSet::new();

        for ival in newhashset.iter() {
            for vali in ival.iter() {
                finalhashset.insert(vali.to_string());
            }
        }

        let mut newvec: Vec<(String, i32)> = Vec::new();

        for i in vecunpack.iter() {
            let unpack = i
                .sequence
                .as_bytes()
                .windows(kmer.parse::<usize>().unwrap())
                .map(|x| str::from_utf8(x).unwrap())
                .collect::<Vec<_>>();
            for i in unpack.iter() {
                for val in finalhashset.iter() {
                    let mut count = 0usize;
                    if i == val {
                        count += 1;
                    }
                    newvec.push((i.to_string(), count as i32));
                }
            }
        }

        let newveccontrol = newvec.iter().map(|x| x.0.clone()).collect::<HashSet<_>>();

        let mut finalcount: HashMap<String, i32> = HashMap::new();

        for newi in newveccontrol.iter() {
            for valnewi in newvec.iter() {
                let mut countseq = 0i32;
                if *newi == valnewi.0 {
                    countseq += valnewi.1;
                }
                finalcount.insert(newi.clone(), countseq);
            }
        }

        let hashses = finalcount
            .iter()
            .map(|val| val.0.to_string())
            .collect::<Vec<_>>();

        /*
        combining the sequence features and the value based for the better prediction
        of the anti-microbial tests.
        */

        let matrix_pre_conversion = convertarray(hashses).unwrap();
        let mut matrix_final_conversion: Vec<Vec<f64>> = Vec::new();

        matrix_pre_conversion
            .iter()
            .for_each(|val| matrix_final_conversion.push(vec![val.parse::<f64>().unwrap()]));

        let sensitivity_values = sensitivity(sensivity).unwrap();

        let matrix_final_convert = matrix_final_conversion
            .iter()
            .map(|x| x[0])
            .collect::<Vec<f64>>();

        let add_sensivity_matrix = matrix_final_convert
            .iter()
            .zip(sensitivity_values)
            .map(|x| vec![x.0 + x.1])
            .collect::<Vec<_>>();

        let density_x = DenseMatrix::from_2d_vec(&add_sensivity_matrix).unwrap();

        Ok(density_x)
    }
}
