
// CPP program to Map Cs and Ds
#include <vector>
#include <random>
#include <cstdlib>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

// Taille des images
const int IMG_SIZE = 15;
const int FLAT_SIZE = IMG_SIZE * IMG_SIZE;

// Étiquettes : +1 pour "C", -1 pour "D"
struct Example {
    vector<int> pixels; // Valeurs des pixels (-1 ou 1)
    int label;          // +1 pour "C", -1 pour "D"
};

void shuffleIndices(vector<int>& indices);
void fillInIndices(int num_train, const vector<int>& indices,vector<int>& even_indices, vector<int>& odd_indices);
void readExamples(int num_train, const vector<int>& even_indices, const vector<int>& odd_indices, vector<Example>& examples);
void initializeWeights(vector<float>& weights, float& bias);

int predict(const vector<int>& pixels, const vector<float>& weights, float bias);
void trainPerceptron(vector<int>& even_indices, vector<int>& odd_indices,
                     const vector<Example>& examples, vector<float>& weights, float& bias, float learningRate, int epochs);

void readTests(int num_train, int num_test, const vector<int>& indices, const vector<float>& weights, float bias);
pair<float, float> findMinMax(const vector<float>& vec);
void dispWeights(const vector<float>& weights, int epoch, int non_zero_errors);

int main(int argc, char* argv[])
{
    // Ratio entraînement / test
    int num_train = 120;
    int num_test  = 20;

    // Mélange des index
    vector<int> indices(num_train + num_test);
    shuffleIndices(indices);

    // Remplissage des indices pairs et impairs
    vector<int> even_indices;
    vector<int> odd_indices;
    fillInIndices(num_train, indices, even_indices, odd_indices);

    // Charger la base d'exemples
    vector<Example> examples;
    readExamples(num_train, even_indices, odd_indices, examples);
    
    // Initialiser les poids et le biais
    vector<float> weights(FLAT_SIZE);
    float bias;
    initializeWeights(weights, bias);
    
    // Entraîner le perceptron
    float learningRate = 0.1f;
    int epochs = 20;
    trainPerceptron(even_indices, odd_indices, examples, weights, bias, learningRate, epochs);
    
    // Tester la prédiction
    readTests(num_train, num_test, indices, weights, bias);
    
    return 0;
}

// Fonction pour initialiser des poids aléatoirement
void initializeWeights(vector<float>& weights, float& bias) {
    for (auto& weight : weights) {
        // weight = static_cast<float>(rand()) / RAND_MAX * 2 - 1; // entre -1 et 1
        weight = static_cast<float>(rand()) / RAND_MAX; // entre 0 et 1
    }
    // bias = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
    bias = static_cast<float>(rand()) / RAND_MAX;
}

void shuffleIndices(vector<int>& indices) {
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);
}

void fillInIndices(int num_train, const vector<int>& indices, vector<int>& even_indices, vector<int>& odd_indices) {
    for (int value : indices)
    {
        if (value % 2 == 0)
        {
            if (even_indices.size() < num_train / 2)
                even_indices.push_back(value);
        }
        else
        {
            if (odd_indices.size() < num_train / 2)
                odd_indices.push_back(value);
        }

        // Stop dès que nous avons assez d'indices
        if (even_indices.size() == num_train / 2 && odd_indices.size() == num_train / 2)
            break;
    }
}

// Fonction de prédiction : calcule la somme pondérée et retourne +1 ou -1
int predict(const vector<int>& pixels, const vector<float>& weights, float bias) {
    float activation = bias;
    for (int i = 0; i < FLAT_SIZE; ++i) {
        activation += weights[i] * pixels[i];
    }
    return activation >= 0 ? 1 : -1;
}

// Entraînement du perceptron
void trainPerceptron(vector<int>& even_indices, vector<int>& odd_indices,
                     const vector<Example>& examples, vector<float>& weights, float& bias, float learningRate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int non_zero_errors = 0;
        dispWeights(weights, epoch, non_zero_errors);

        for (int i = 0; i < examples.size(); ++i) {
            Example example = examples[i];
            int prediction = predict(example.pixels, weights, bias);
            int error = example.label - prediction;
            if (error != 0) { // Mise à jour si erreur
                non_zero_errors += 1;
                for (int j = 0; j < FLAT_SIZE; ++j) {
                    weights[j] += learningRate * error * example.pixels[j];
                }
                bias += learningRate * error;
                dispWeights(weights, epoch, non_zero_errors);
            }
        }
        cout << "Epoch #" << setw(2) << epoch << ", err = " << non_zero_errors << " / " << examples.size() << endl;
    }
}

void readExamples(int num_train, const vector<int>& even_indices, const vector<int>& odd_indices, vector<Example>& examples) {
    int l = -1;
    for (size_t k = 0; k < num_train; ++k) {
        if (k % 2 == 0) {
            // it's even so it's a "C"
            l = even_indices[k/2];
            // cout << ". Reading C image " << l << "..." << endl;
        }
        else {
            // it's odd so it's a "D"
            l = odd_indices[(k-1)/2];
            // cout << ". Reading D image " << l << "..." << endl;
        }

        Mat roi = imread("../img/binning/roi_nxn_" + to_string(l) + ".jpg", IMREAD_GRAYSCALE);
        // cout << "ROI_" << k << " = " << roi << endl;
        
        Example example;
        if (l % 2 == 0) {
            example.label = 1; // "C"
        } else {
            example.label = -1; // "D"
        }

        int pixel;
        example.pixels = vector<int>(FLAT_SIZE, 0);

        for (int i = 0; i < roi.rows; ++i) {
            for (int j = 0; j < roi.cols; ++j) {
                if (roi.at<uchar>(i, j) < 128) {
                    pixel = 1; // Black
                }
                else {
                    pixel = -1; // White
                }
                example.pixels[i * IMG_SIZE + j] = pixel;
            }
        }

        examples.push_back(example);
    }
}

void readTests(int num_train, int num_test, const vector<int>& indices, const vector<float>& weights, float bias) {
    int num_OK = 0;
    int num_KO = 0;

    for (size_t k = num_train; k < num_train + num_test; ++k) {
        int l = indices[k];
        Mat roi = imread("../img/binning/roi_nxn_" + to_string(l) + ".jpg", IMREAD_GRAYSCALE);
        // cout << "ROI_" << k << " = " << roi << endl;
        
        Example test;
        if (l % 2 == 0) {
            test.label = 1; // "C"
        } else {
            test.label = -1; // "D"
        }

        int pixel;
        test.pixels = vector<int>(FLAT_SIZE, 0);

        for (int i = 0; i < roi.rows; ++i) {
            for (int j = 0; j < roi.cols; ++j) {
                if (roi.at<uchar>(i, j) < 128) {
                    pixel = 1; // Black
                }
                else {
                    pixel = -1; // White
                }
                test.pixels[i * IMG_SIZE + j] = pixel;
            }
        }

        cout << "Prediction #" << setw(3) << l << " for " << (test.label == 1 ? "C: " : "D: ") <<
                                     (predict(test.pixels, weights, bias) == 1 ? "C" : "D") << 
                            " ~> " << (test.label == predict(test.pixels, weights, bias) ? "OK !" : "KO !") << endl;
        
        if (predict(test.pixels, weights, bias) == test.label) {
            num_OK += 1;
        }
        else {
            num_KO += 1;
        }
    }

    cout << ". Final score = " << num_OK << "/" << num_OK + num_KO << endl;
}

pair<float, float> findMinMax(const vector<float>& vec) {
    if (vec.empty()) {
        throw runtime_error("Le vecteur est vide.");
    }
    auto minIt = min_element(vec.begin(), vec.end());
    auto maxIt = max_element(vec.begin(), vec.end());
    return { *minIt, *maxIt };
}

void dispWeights(const vector<float>& weights, int epoch, int non_zero_errors) {
    auto [minVal, maxVal] = findMinMax(weights);
    // cout << "Minimum: " << minVal << endl;
    // cout << "Maximum: " << maxVal << endl;
    Mat weights_image = Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8U);

    unsigned int weight_pixel;
    for (int k = 0; k < FLAT_SIZE; ++k) {
        weight_pixel = 255 * ((weights[k] - minVal) / (maxVal - minVal));
        int i = k / IMG_SIZE;
        int j = k % IMG_SIZE;
        // cout << "[" << i << "," << j << "] ~> " << weight_pixel << endl;
        weights_image.at<uchar>(i, j) = weight_pixel;
    }
    imwrite("../img/weights/weights_" + to_string(epoch + non_zero_errors) + ".jpg", weights_image);
}