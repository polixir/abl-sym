local train_data_path = "./data/data.npz";
local validation_data_path = "./data/interpolate_sample.npy";
local batch_size = 128;
local cuda_device = [0, 1, 2, 3, 4, 5, 6, 7];
local num_epochs = 200;
local pretrain_file = '';

{
  "dataset_reader": {
    "type": "dm_math",
    "with_expr_anno": false,
    "with_program_anno": true,
    "target_key": "ans",
    "tokenizer": {
      "ques": {
        "type": "character"
      },
      "ans":  {
        "type": "character"
      }
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      },
    },
    "target_token_indexers": {
      "answer": {
        "type": "single_id",
        "namespace": "answer",
        "start_tokens": ["@start@"],
        "end_tokens": ["@end@"],
      },
      "program": {
        "type": "single_id",
        "namespace": "program",
        "start_tokens": ["@start@"],
        "end_tokens": ["@end@"],
      }
    },
  },
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "vocabulary": {
    "directory_path": "./data/v1.0/mix/vocabulary",
//    "tokens_to_add": {
//        "target_tokens": ["@start@", "@end@"],
//    },
  },
  "model": {
    "type": "mix-transformer",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 512,
          "trainable": false
        },
      }
    },
    "target_embedders": {
      "answer": {
          "token_embedders": {
            "answer": {
              "type": "embedding",
              "vocab_namespace": "answer",
              "embedding_dim": 512,
              "trainable": false
            },
          },
      },
      "program": {
          "token_embedders": {
            "program": {
              "type": "embedding",
              "vocab_namespace": "program",
              "embedding_dim": 512,
              "trainable": false
            },
          }
      },
    },
    "transformer": {
       "d_model": 512,
       "nhead": 8,
       'num_encoder_layers': 6,
       'num_decoder_layers': {
            "answer": 6,
            "program": 2,
       },
       'dim_feedforward': 2048,
       'dropout': 0.1,
       'activation': "relu"
    },
    "loss_coefs": {
        "answer": 0.5,
        "program": 0.5,
    },
    "max_decoding_steps": 31,
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : batch_size,
    "sorting_keys": [["source_tokens", "num_tokens"], ["target_tokens", "num_tokens"]]
  },
  "trainer": {
    "type": "pt_dist",
    "num_epochs": num_epochs,
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
      "betas": [0.9, 0.995],
      "eps": 1e-9,
      "parameter_groups": [
           [["encoder", "decoder"], {"lr": 1e-4}],
      ],
    },
    "grad_clipping": 0.1,
    "cuda_device": cuda_device,
    "learning_rate_scheduler": {
        "type": "multi_step",
        "milestones": [5, 10],
        "gamma": 0.3
    },
    "validation_metric": "+answer_acc",
    "should_log_parameter_statistics": false,
    "should_log_learning_rate": true,
    "pretrain_file": pretrain_file,
  }
}
