use crate::structadd::AddGene;
use crate::structadd::Predict;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};
use std::error::Error;
use std::fs::File;
use std::io::Write;

/*
Gaurav Sablok
codeprog@icloud.com
*/

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

const LEARNING_RATE: f64 = 1e-3;
const EPOCHS: usize = 100;
const BATCH_SIZE: usize = 32;

// Backend selection. NdArray runs on CPU; swap for burn::backend::Wgpu (or
// burn::backend::Cuda / burn::backend::LibTorch, depending on which backend
// feature you enable in Cargo.toml) to move training onto a GPU.
pub type InnerBackend = NdArray<f32>;
pub type TrainBackend = Autodiff<InnerBackend>;

// ---------------------------------------------------------------------------
// Logistic Regression model
// ---------------------------------------------------------------------------

/// A single linear layer followed by a sigmoid (binary) or softmax
/// (multi-class) activation — the canonical logistic regression.
#[derive(Module, Debug)]
pub struct LogisticRegression<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LogisticRegression<B> {
    /// Build the model.
    ///
    /// * `in_dim`  – number of input features (kmer dimensionality)
    /// * `out_dim` – number of classes (2 for binary, N for multi-class)
    fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(in_dim, out_dim).init(device);
        Self { linear }
    }

    /// Forward pass — returns raw logits (no activation applied here;
    /// `CrossEntropyLoss` expects logits, not probabilities).
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }

    /// Convenience: sigmoid probabilities for binary classification.
    #[allow(dead_code)]
    fn predict_proba_binary(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::sigmoid(self.forward(x))
    }

    /// Convenience: softmax probabilities for multi-class classification.
    #[allow(dead_code)]
    fn predict_proba_multiclass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::softmax(self.forward(x), 1)
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Compute classification accuracy given integer label tensors.
fn accuracy<B: Backend>(predictions: Tensor<B, 2>, labels: Tensor<B, 1, Int>) -> f64 {
    // argmax over class dimension → predicted class index, shape [n, 1] -> [n]
    let pred_classes: Tensor<B, 1, Int> = predictions.argmax(1).squeeze::<1>();
    let total = labels.dims()[0] as f64;
    let correct: f32 = pred_classes.equal(labels).int().sum().into_scalar().elem();
    correct as f64 / total
}

fn train_test_split<B: Backend>(
    x: Tensor<B, 2>,
    y: Tensor<B, 1, Int>,
    test_ratio: f64,
) -> (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 1, Int>,
    Tensor<B, 1, Int>,
) {
    let n = x.dims()[0];
    let n_test = (n as f64 * test_ratio).round() as usize;
    let n_train = n - n_test;

    // Simple sequential split — shuffle upstream if randomness is needed.
    let x_train = x.clone().narrow(0, 0, n_train);
    let x_test = x.narrow(0, n_train, n_test);
    let y_train = y.clone().narrow(0, 0, n_train);
    let y_test = y.narrow(0, n_train, n_test);

    (x_train, x_test, y_train, y_test)
}

/// Mini-batch iterator: yields (x_batch, y_batch) slices.
fn minibatches<B: Backend>(
    x: &Tensor<B, 2>,
    y: &Tensor<B, 1, Int>,
    batch_size: usize,
) -> Vec<(Tensor<B, 2>, Tensor<B, 1, Int>)> {
    let n = x.dims()[0];
    let mut batches = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + batch_size).min(n);
        let xb = x.clone().narrow(0, start, end - start);
        let yb = y.clone().narrow(0, start, end - start);
        batches.push((xb, yb));
        start = end;
    }
    batches
}

pub fn burn_run(
    pathfile: &str,
    predictfile: &str,
    kmers: &str,
    sensivitythreshold: &str,
    sensitivityfile: &str,
) -> Result<String, Box<dyn Error>> {
    // ------------------------------------------------------------------
    // 1. Build feature matrices  (reuse your existing pipeline)
    // ------------------------------------------------------------------
    let fastafile = AddGene {
        fastafile: pathfile.to_string(),
    };
    let predictionfile = Predict {
        predictionfile: predictfile.to_string(),
    };

    // fastamatrix.0 → 2-D f32 array of shape [n_samples, n_features]
    // fastamatrix.1 → 1-D u32 array of shape [n_samples]  (class labels)
    let fastamatrix = fastafile
        .generatemers(kmers, sensivitythreshold, sensitivityfile)
        .unwrap();
    let predictionmatrix = predictionfile
        .generatemers(kmers, sensivitythreshold)
        .unwrap();

    // ------------------------------------------------------------------
    // 2. Set up the Burn device
    // ------------------------------------------------------------------
    let device = Default::default();

    // Convert your ndarray / Vec<Vec<f32>> into flat Vecs, matching whatever
    // type `generatemers` returns.
    let n_samples = fastamatrix.0.iter().collect::<Vec<_>>().len();
    let n_features = fastamatrix.0.iter().collect::<Vec<_>>().len();
    let n_classes = *fastamatrix.1.iter().max().unwrap() as usize + 1;

    // Flatten 2-D feature matrix into a contiguous f32 Vec
    let x_flat: Vec<f32> = fastamatrix
        .0
        .iter()
        .collect::<Vec<_>>()
        .iter()
        .map(|x| x.to_string().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let y_flat: Vec<i32> = fastamatrix
        .1
        .iter()
        .map(|x| x.to_string().parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    let x_all: Tensor<TrainBackend, 2> =
        Tensor::<TrainBackend, 1>::from_floats(x_flat.as_slice(), &device)
            .reshape([n_samples, n_features]);
    let y_all: Tensor<TrainBackend, 1, Int> =
        Tensor::<TrainBackend, 1, Int>::from_ints(y_flat.as_slice(), &device);

    // Prediction matrix
    let pred_n = predictionmatrix.iter().collect::<Vec<_>>().len();
    let pred_flat: Vec<f32> = predictionmatrix
        .iter()
        .collect::<Vec<_>>()
        .iter()
        .map(|x| x.to_string().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let x_pred: Tensor<InnerBackend, 2> =
        Tensor::<InnerBackend, 1>::from_floats(pred_flat.as_slice(), &device)
            .reshape([pred_n, n_features]);

    // ------------------------------------------------------------------
    // 3. Train / test split
    // ------------------------------------------------------------------
    let (x_train, x_test, y_train, y_test) = train_test_split(x_all, y_all, 0.2);

    // ------------------------------------------------------------------
    // 4. Build model + optimiser
    // ------------------------------------------------------------------
    let mut model: LogisticRegression<TrainBackend> =
        LogisticRegression::new(n_features, n_classes, &device);
    let mut optim = AdamConfig::new().init();

    // ------------------------------------------------------------------
    // 5. Training loop
    // ------------------------------------------------------------------
    println!("Starting logistic regression training…");
    println!(
        "  Samples: {}  |  Features: {}  |  Classes: {}",
        n_samples, n_features, n_classes
    );

    for epoch in 0..EPOCHS {
        let batches = minibatches(&x_train, &y_train, BATCH_SIZE);
        let n_batches = batches.len();
        let mut epoch_loss = 0f32;

        for (xb, yb) in batches {
            // Forward pass
            let logits = model.forward(xb);

            // Cross-entropy loss (Burn expects [batch, classes] logits and
            // [batch] integer labels)
            let loss = CrossEntropyLossConfig::new()
                .init(&device)
                .forward(logits, yb);

            epoch_loss += loss.clone().into_scalar().elem::<f32>();

            // Backward pass + parameter update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);
        }

        // Validation accuracy every 10 epochs
        if (epoch + 1) % 10 == 0 {
            let test_logits = model.forward(x_test.clone());
            let acc = accuracy(test_logits, y_test.clone());
            println!(
                "Epoch {:>4}/{} — avg loss: {:.4}  val accuracy: {:.4}",
                epoch + 1,
                EPOCHS,
                epoch_loss / n_batches as f32,
                acc
            );
        }
    }

    // ------------------------------------------------------------------
    // 6. Final evaluation on the held-out test set
    // ------------------------------------------------------------------
    let test_logits = model.forward(x_test.clone());
    let final_accuracy = accuracy(test_logits, y_test.clone());
    println!("Final model accuracy: {:.4}", final_accuracy);

    // ------------------------------------------------------------------
    // 7. Predict on new sequences
    // ------------------------------------------------------------------
    // Move the trained weights onto the inference (non-autodiff) backend.
    let inference_model: LogisticRegression<InnerBackend> = model.valid();
    let pred_logits = inference_model.forward(x_pred);
    // Predicted class index for each new sequence
    let pred_classes: Tensor<InnerBackend, 1, Int> = pred_logits.argmax(1).squeeze::<1>();
    let pred_vec: Vec<i32> = pred_classes.into_data().convert::<i32>().to_vec().unwrap();

    // ------------------------------------------------------------------
    // 8. Write predictions to disk  (same pattern as the original knnrun)
    // ------------------------------------------------------------------
    let mut filewrite = File::create("logistic-regression-predict")?;
    writeln!(
        filewrite,
        "Logistic regression prediction values are as follows"
    )?;
    for class_idx in &pred_vec {
        writeln!(filewrite, "{}", class_idx)?;
    }

    Ok(format!(
        "Logistic regression trained — final accuracy: {:.4}",
        final_accuracy
    ))
}
