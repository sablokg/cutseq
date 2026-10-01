# cutseq

- Logistic regression modelling of the sensitivity assay for patients along with the sequencing 
- A new approach to combine the sequence information to the sensitivity assays.
- First generate the kmer plot to see how much threshold you want to put for the deep learning. 
- All classifiers under one hood algorithmic based to neural based classifier. 

```
cargo build

```

```
                 _                        
   ___   _   _  | |_   ___    ___    __ _ 
  / __| | | | | | __| / __|  / _ \  / _` |
 | (__  | |_| | | |_  \__ \ |  __/ | (_| |
  \___|  \__,_|  \__| |___/  \___|  \__, |
                                       |_|

Hash Regression based identification of antimicrobial
       ************************************************
       Author Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************

Usage: cutseq <COMMAND>

Commands:
  kmerplot        Generate kmer plot
  microbial-log   Logistic Antimicrobial
  elastic-net     Elastic Net
  random-forest   Random Forest for classification
  knn-classifier  KNN Classifier for classification
  regression      Candle regression using transformers
  help            Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```

