use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn convertarray(pathseq: Vec<String>) -> Result<Vec<String>, Box<dyn Error>> {
    let arrayclone = pathseq.clone();
    let mut returnvec: Vec<String> = Vec::new();

    for i in arrayclone.iter() {
        let stringcut = i.chars().collect::<Vec<_>>();
        for i in stringcut.iter() {
            let mut returnstring: Vec<f64> = Vec::new();
            if *i == 'A' {
                returnstring.push(0f64);
            } else if *i == 'T' {
                returnstring.push(1f64);
            } else if *i == 'G' {
                returnstring.push(2f64);
            } else if *i == 'C' {
                returnstring.push(3f64);
            } else {
                continue;
            }
            let itercombine = returnstring
                .iter()
                .map(|val| val.to_string())
                .collect::<Vec<_>>()
                .concat()
                .to_string();
            returnvec.push(itercombine)
        }
    }

    Ok(returnvec)
}
