//
// Created by fss on 24-2-15.
//
#include <cstdio>
#include <cstring>
#include <ctime>
#include "llama_chat.hpp"
int main(int argc, char* argv[]) {
  // default parameters
  char* checkpoint_path = "tmp/llama2/llama2_7b.bin";  // e.g. out/model.bin
  char* tokenizer_path = "tmp/llama2/tokenizer.bin";
  float temperature = 1.0f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;         // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;           // number of steps to run for
  char* prompt = NULL;       // prompt string
  unsigned long long rng_seed = 0;  // seed rng with time by default
  char* mode = "generate";          // generate|chat
  // poor man's C argparse so we can override the defaults above from the command line

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len;  // ovrerride to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps, false);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}