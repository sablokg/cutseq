use crate::structadd::Fasta;
use kuva::backend::svg::SvgBackend;
use kuva::plot::BarPlot;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;
use kuva::render::render::render_multiple;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn kmerplot(pathfile: &str, kmerstradd: &str) -> Result<String, Box<dyn Error>> {
    let value = File::open(pathfile).expect("file not present");
    let valueread = BufReader::new(value);

    let mut header: Vec<String> = Vec::new();
    let mut sequence: Vec<String> = Vec::new();

    for i in valueread.lines() {
        let line = i.expect("value not present");
        if line.starts_with(">") {
            let linemerge = line.split(">").map(|x| x.to_string()).collect::<Vec<_>>();
            header.push(linemerge[0].clone());
        }
        if !line.starts_with(">") {
            sequence.push(line);
        }
    }

    let fastapack = merge(header, sequence).unwrap();

    let kmerstr = fastapack
        .iter()
        .map(|x| x.sequence.clone())
        .collect::<Vec<_>>();

    let mut kmerfreq: HashSet<String> = HashSet::new();

    for i in kmerstr.iter() {
        let seqval = i
            .as_bytes()
            .windows(kmerstradd.parse::<usize>().unwrap())
            .map(|x| str::from_utf8(x).unwrap())
            .collect::<Vec<_>>();
        for val in seqval.iter() {
            kmerfreq.insert(val.to_string());
        }
    }

    let mut counthashes: Vec<(String, f64)> = Vec::new();

    for i in kmerstr.iter() {
        for valseq in kmerfreq.iter() {
            let seqval = i
                .as_bytes()
                .windows(kmerstradd.parse::<usize>().unwrap())
                .map(|x| str::from_utf8(x).unwrap())
                .collect::<Vec<_>>();
            for iterseq in seqval.iter() {
                let mut countseq = 0f64;
                if iterseq == valseq {
                    countseq += 1f64;
                }
                counthashes.push((iterseq.to_string(), countseq));
            }
        }
    }

    let plot = BarPlot::new()
        .with_bars(counthashes)
        .with_color("steelblue");

    let plots = vec![Plot::Bar(plot)];
    let layout = Layout::auto_from_plots(&plots)
        .with_title("kmer frequencies")
        .with_y_label("Count");

    let scene = render_multiple(plots, layout);
    let svg = SvgBackend.render_scene(&scene);
    std::fs::write("bar.svg", svg).unwrap();

    Ok("The path has been written".to_string())
}

pub fn merge(pathfasta: Vec<String>, pathseq: Vec<String>) -> Result<Vec<Fasta>, Box<dyn Error>> {
    let mut returnfasta: Vec<Fasta> = Vec::new();

    for i in 0..pathfasta.len() {
        returnfasta.push(Fasta {
            header: pathfasta[i].clone(),
            sequence: pathseq[i].clone(),
        })
    }

    Ok(returnfasta)
}
