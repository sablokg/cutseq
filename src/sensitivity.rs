use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn sensitivity(pathfile: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("path not present");
    let fileread = BufReader::new(fileopen);

    let mut vecadd: Vec<f64> = Vec::new();

    for i in fileread.lines() {
        let line = i.expect("path line not found");
        vecadd.push(line.parse::<f64>().unwrap());
    }
    Ok(vecadd)
}
