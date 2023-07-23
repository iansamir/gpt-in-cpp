#include <torch/torch.h>
#include <iostream>
#include <fstream>

#include <map>
#include <sstream>
#include <vector>

// Define the NewGELU Module
struct NewGELU : torch::nn::Module {
    torch::Device device = torch::kCPU; // Default device

    // Constructor
    NewGELU() {
        // Use a GPU if available
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // Ensure the input tensor is on the correct device and is a floating-point tensor
        x = x.to(device).to(torch::kFloat);

        return 0.5 * x * (1.0 + torch::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * torch::pow(x, 3.0))));
    }
};

// Define the CausalSelfAttention Module
struct CausalSelfAttention : torch::nn::Module {
    int n_head, n_embd;
    torch::nn::Linear c_attn{nullptr}, c_proj{nullptr};
    torch::nn::Dropout attn_dropout{nullptr}, resid_dropout{nullptr};

    CausalSelfAttention(int n_embd, int n_head)
        : n_head(n_head), n_embd(n_embd) {
        // ensure the dimensionality division is exact
        TORCH_CHECK(n_embd % n_head == 0);

        c_attn = register_module("c_attn", torch::nn::Linear(n_embd, 3 * n_embd));
        c_proj = register_module("c_proj", torch::nn::Linear(n_embd, n_embd));

        attn_dropout = register_module("attn_dropout", torch::nn::Dropout(0.1));
        resid_dropout = register_module("resid_dropout", torch::nn::Dropout(0.1));
    }

    torch::Tensor forward(torch::Tensor x) {
        int B = x.size(0);
        int T = x.size(1);
        int C = x.size(2);

        // calculate query, key, values for all heads in batch
        auto tmp = c_attn->forward(x).view({B, T, 3, n_head, n_embd/n_head});
        auto q = tmp.select(2, 0);
        auto k = tmp.select(2, 1);
        auto v = tmp.select(2, 2);

        // perform dot-product attention
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);
        auto att_score = q.matmul(k.transpose(-2, -1)) / std::sqrt(k.size(-1));
        auto att_prob = torch::softmax(att_score, -1);
        att_prob = attn_dropout->forward(att_prob);
        auto y = att_prob.matmul(v);
        y = y.transpose(1, 2).contiguous().view({B, T, C});
        
        // output projection
        y = resid_dropout->forward(c_proj->forward(y));
        return y;
    }
};

// Define the Block Module
struct Block : torch::nn::Module {
    torch::nn::LayerNorm ln_1{nullptr}, ln_2{nullptr};
    CausalSelfAttention attn{nullptr};
    torch::nn::Linear c_fc{nullptr}, c_proj{nullptr};
    torch::nn::Dropout dropout{nullptr};
    NewGELU gelu{nullptr};

    Block(int n_embd) {
        ln_1 = register_module("ln_1", torch::nn::LayerNorm(n_embd));
        attn = register_module("attn", CausalSelfAttention(n_embd));
        ln_2 = register_module("ln_2", torch::nn::LayerNorm(n_embd));
        c_fc = register_module("c_fc", torch::nn::Linear(n_embd, 4 * n_embd));
        c_proj = register_module("c_proj", torch::nn::Linear(4 * n_embd, n_embd));
        dropout = register_module("dropout", torch::nn::Dropout(0.1));
        gelu = register_module("gelu", NewGELU());
    }

    torch::Tensor forward(torch::Tensor x) {
        auto a = attn->forward(ln_1->forward(x));
        x = x + a;
        auto mlp = dropout->forward(c_proj->forward(gelu->forward(c_fc->forward(ln_2->forward(x)))));
        x = x + mlp;
        return x;
    }
};

// Define the GPT Module
struct GPT : torch::nn::Module {
    torch::nn::Embedding wte{nullptr}, wpe{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::ModuleList h;
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};

    GPT(int vocab_size, int block_size, int n_layer, int n_embd) {
        // Embeddings
        wte = register_module("wte", torch::nn::Embedding(vocab_size, n_embd));
        wpe = register_module("wpe", torch::nn::Embedding(block_size, n_embd));
        drop = register_module("drop", torch::nn::Dropout(0.1));
        ln_f = register_module("ln_f", torch::nn::LayerNorm(n_embd));
        lm_head = register_module("lm_head", torch::nn::Linear(n_embd, vocab_size, false));

        // Transformer blocks
        for(int i = 0; i < n_layer; i++) {
            h.push_back(std::make_shared<Block>(n_embd)); // Assuming Block is implemented
            register_module("h_" + std::to_string(i), h[i]);
        }
    }

    torch::Tensor forward(torch::Tensor idx, torch::Tensor targets = {}) {
        auto device = idx.device();
        auto b = idx.size(0);
        auto t = idx.size(1);
        auto pos = torch::arange(0, t, torch::kLong, device).unsqueeze(0);

        // forward the GPT model itself
        auto tok_emb = wte->forward(idx);
        auto pos_emb = wpe->forward(pos);
        auto x = drop->forward(tok_emb + pos_emb);

        for(auto& block : h) {
            x = block->as<Block>()->forward(x); // Cast Module to Block
        }

        x = ln_f->forward(x);
        auto logits = lm_head->forward(x);

        // if we are given some desired targets also calculate the loss
        torch::Tensor loss = {};
        if (targets.defined()) {
            loss = torch::nn::functional::cross_entropy(logits.view({-1, logits.size(-1)}), targets.view(-1), {}, -1);
        }

        return logits; // Loss could also be returned if needed
    }
};

std::vector<std::vector<int64_t>> load_data(const std::string& filename) {
    std::vector<std::string> sentences;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        sentences.push_back(line);
    }

    return tokenize(sentences);
}

// Simple tokenizer that splits sentences into words and maps each word to a unique integer ID
std::vector<std::vector<int64_t>> tokenize(const std::vector<std::string>& sentences) {
    std::map<std::string, int64_t> word_to_id;
    std::vector<std::vector<int64_t>> tokenized_sentences;

    for (const auto& sentence : sentences) {
        std::vector<int64_t> tokens;
        std::istringstream iss(sentence);
        std::string word;

        while (iss >> word) {
            if (word_to_id.find(word) == word_to_id.end()) {
                word_to_id[word] = word_to_id.size();  // Assign a new ID to a new word
            }

            tokens.push_back(word_to_id[word]);
        }

        tokenized_sentences.push_back(tokens);
    }

    return tokenized_sentences;
}


// Function to generate batches of data and pad sequences within a batch to the same length
std::vector<std::vector<std::vector<int64_t>>> make_batches(const std::vector<std::vector<int64_t>>& data, size_t batch_size) {
    std::vector<std::vector<std::vector<int64_t>>> batches;

    for (size_t i = 0; i < data.size(); i += batch_size) {
        std::vector<std::vector<int64_t>> batch(data.begin() + i, data.begin() + std::min(data.size(), i + batch_size));
        
        // Find the length of the longest sequence in the current batch
        size_t max_len = 0;
        for (const auto& seq : batch) {
            if (seq.size() > max_len) {
                max_len = seq.size();
            }
        }

        // Pad all sequences in the batch to the length of the longest sequence
        for (auto& seq : batch) {
            seq.resize(max_len, 0);  // Pad with zeros
        }

        batches.push_back(batch);
    }

    return batches;
}

int main() {
    torch::Device device = torch::kCUDA;
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA not available, reverting to CPU." << std::endl;
        device = torch::kCPU;
    }

    const int vocab_size = 1000;
    const int block_size = 512;
    const int n_layer = 6;
    const int n_embd = 768;
    const int num_epochs = 5;
    const int batch_size = 2;
    const int seq_length = 10;

    // Load and process data
    std::vector<std::vector<int64_t>> data = load_data("arlin_transcripts.txt");
    std::vector<std::vector<std::vector<int64_t>>> batches = make_batches(data, batch_size);

    // Instantiate the GPT model
    GPT model(vocab_size, block_size, n_layer, n_embd);
    model.to(device);

    // Define an optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (const auto& batch : batches) {
            // Convert batch to tensor and send to device
            torch::Tensor idx = torch::from_blob(batch.data(), {batch.size(), seq_length}, torch::kLong).to(device);
            torch::Tensor targets = idx.clone();  // In this example, we just use input as targets

            // Run the model
            auto [logits, loss] = model.forward(idx, targets);

            // Perform a backward pass on the loss
            loss.backward();

            // Step the optimizer and zero the gradient for the next iteration
            optimizer.step();
            optimizer.zero_grad();

            // Print the loss for this epoch
            std::cout << "Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Let's do a prediction now. This should be done with torch::no_grad context
    torch::Tensor input = torch::randint(vocab_size, {1, seq_length}, device);
    torch::NoGradGuard no_grad;
    auto [logits, _] = model.forward(input);
    auto prediction = logits.argmax(2);
    std::cout << "Input: " << input << std::endl;
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
