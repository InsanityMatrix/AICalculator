package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
)

type RawAI struct {
	InputLayer   []float64   `json:"input_layer"`
	HiddenLayer  []float64   `json:"hidden_layer"`
	OutputLayer  []float64   `json:"output_layer"`
	WeightsIH    [][]float64 `json:"weights_ih"`
	WeightsHO    [][]float64 `json:"weights_ho"`
	LearningRate float64     `json:"learning_rate"`
}

func NewRawAI(inputSize, hiddenSize, outputSize int, learningRate float64) *RawAI {
	nn := RawAI{
		InputLayer:   make([]float64, inputSize),
		HiddenLayer:  make([]float64, hiddenSize),
		OutputLayer:  make([]float64, outputSize),
		WeightsIH:    randomMatrix(inputSize, hiddenSize),
		WeightsHO:    randomMatrix(hiddenSize, outputSize),
		LearningRate: learningRate,
	}
	return &nn
}
func (nn *RawAI) FeedForward(inputs []float64) []float64 {
	// Set input layer
	for i := 0; i < len(inputs); i++ {
		nn.InputLayer[i] = inputs[i]
	}

	// Compute hidden layer
	for i := 0; i < len(nn.HiddenLayer); i++ {
		sum := 0.0
		for j := 0; j < len(nn.InputLayer); j++ {
			sum += nn.InputLayer[j] * nn.WeightsIH[j][i]
		}
		nn.HiddenLayer[i] = sum
		if math.IsNaN(sum) {
			panic(fmt.Sprintf("Sum is NaN on Hidden Layer:\nInput Layer: %v\nHidden Layer: %v\nWeights IH: %v\n", nn.InputLayer, nn.HiddenLayer, nn.WeightsIH))
		}

	}

	// Compute output layer
	for k := 0; k < len(nn.OutputLayer); k++ {
		sum := 0.0
		for j := 0; j < len(nn.HiddenLayer); j++ {
			sum += nn.HiddenLayer[j] * nn.WeightsHO[j][k]
		}
		nn.OutputLayer[k] = sum
		if math.IsNaN(sum) {
			panic(fmt.Sprintf("Sum is NaN on Output Layer:\n Model: %v\n", nn))
		}

	}

	return nn.OutputLayer
}
func (nn *RawAI) Train(inputs []float64, targets []float64) {
	nn.FeedForward(inputs)

	// Compute output layer error
	outputErrors := make([]float64, len(targets))
	for k := 0; k < len(targets); k++ {
		//outputErrors[k] = -1 * (targets[k] - nn.OutputLayer[k])
		outputErrors[k] = nn.OutputLayer[k] - targets[k]
	}

	// Compute hidden layer error
	hiddenErrors := make([]float64, len(nn.HiddenLayer))
	for j := 0; j < len(nn.HiddenLayer); j++ {
		errorSum := 0.0
		for k := 0; k < len(nn.OutputLayer); k++ {
			errorSum += outputErrors[k] * nn.WeightsHO[j][k]
		}
		hiddenErrors[j] = errorSum * sigmoidDerivative(nn.HiddenLayer[j])
		if math.IsInf(math.Abs(hiddenErrors[j]), 1) {
			//Find out why
			fmt.Printf("Hidden Error is Infinite:\nTargets:%v\nOutputLayer:%v\n\n", targets, nn.OutputLayer)
		}
	}

	// Update weights
	for j := 0; j < len(nn.HiddenLayer); j++ {
		for k := 0; k < len(nn.OutputLayer); k++ {
			delta := nn.LearningRate * outputErrors[k] * nn.HiddenLayer[j]
			nn.WeightsHO[j][k] += delta
		}
	}
	for i := 0; i < len(nn.InputLayer); i++ {
		for j := 0; j < len(nn.HiddenLayer); j++ {
			delta := nn.LearningRate * hiddenErrors[j] * nn.InputLayer[i]
			if delta > 0 {
			}
			nn.WeightsIH[i][j] += delta
			if math.IsNaN(delta) {
				fmt.Print(fmt.Sprintf("Delta is NaN.\n Learning Rate: %f\nHidden Errors: %f\nInput: %f\n", nn.LearningRate, hiddenErrors[j], nn.InputLayer[i]))
			}
			if math.IsNaN(nn.WeightsIH[i][j]) {
				fmt.Print(fmt.Sprintf("Delta is NaN.\n Learning Rate: %f\nHidden Errors: %f\nInput: %f\n", nn.LearningRate, hiddenErrors[j], nn.InputLayer[i]))
			}
		}
	}

}
func (nn *RawAI) ExportWeights(filename string) error {
	weightsJson, err := json.Marshal(nn)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(filename, weightsJson, 0644)
	if err != nil {
		return err
	}
	return nil
}
func (nn *RawAI) ImportWeights(filename string) error {
	weightsJson, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	err = json.Unmarshal(weightsJson, nn)
	if err != nil {
		return err
	}
	return nil
}

//RawAI Tools:
func randomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = 1.0
		}
	}
	return matrix
}
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + exp(-x))
}
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func exp(x float64) float64 {
	return 1.0 + x + (x*x)/2.0 + (x*x*x)/6.0 + (x*x*x*x)/24.0
}
