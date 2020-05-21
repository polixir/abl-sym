local NUM_GPUS = 4;
local NUM_THREADS = 23;

local BASE_READER = {
    "type": "dm_math",
    "with_expr_anno": false,
    "with_program_anno": true,
    "target_key": "anno",
    "tokenizer": {
      "ques": {
        "type": "dm_math"
      },
      "ans":  {
        "type": "character"
      }
    },
    "source_token_indexers": {
      "tokens": {
        "type": "math_single_id",
        "namespace": "source_tokens"
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
      },
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens",
        "start_tokens": ["@start@"],
        "end_tokens": ["@end@"],
      }
    },
};
local BASE_ITERATOR = {
  "type": "bucket",
  "batch_size": 256,
  "sorting_keys": [["source_tokens", "num_tokens"], ["target_tokens", "num_tokens"]],
};

{
  "dataset_reader": {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 4096,
    "epochs_per_read": 1,
  },
  "train_data_path": "./data/qp/splits/*",
  "validation_data_path": "./data/qp/interpolate_sample.txt",
  "vocabulary": {
    "directory_path": "./data/qp/mix-base-vocab",
  },
  "model": {
    "type": "my-transformer",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 256,
          "trainable": false
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": 64,
            },
            "encoder": {
                "type": "cnn",
                "embedding_dim": 64,
                "num_filters": 256,
                "ngram_filter_sizes": [5],
            },
            "dropout": 0.1,
        },
      }
    },
    "target_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "target_tokens",
          "embedding_dim": 512,
          "trainable": false
        },
      }
    },
    "transformer": {
       "d_model": 512,
       "nhead": 8,
       'num_encoder_layers': 6,
       'num_decoder_layers': 2,
       'dim_feedforward': 2048,
       'dropout': 0.1,
       'activation': "relu"
    },
    "max_decoding_steps": 21,
    "target_namespace": "target_tokens",
    "use_bleu": false,
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    "output_queue_size": 4096,
  },
  "trainer": {
    "num_epochs": 10,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001,
      "betas": [0.9, 0.995],
      "eps": 1e-9,
    },
    "grad_clipping": 0.1,
    "cuda_device": [0, 1, 2, 3],
    "learning_rate_scheduler": {
        "type": "multi_step",
        "milestones": [5, 10, 20],
        "gamma": 0.3
    },
    "validation_metric": "+seq_acc",
    "should_log_parameter_statistics": false,
    "should_log_learning_rate": true,
  }
}
