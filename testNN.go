package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func XORNeuralNetworkTest() {
	g := NewGraph()
	mind := newBrain(g)

	inputVals := []float64{1, 0, 0, 1, 1, 1, 0, 0}
	inputTensor := tensor.New(tensor.WithBacking(inputVals), tensor.WithShape(4, 2))
	input := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(4, 2),
		WithValue(inputTensor),
	)

	//Output Set
	outputVals := []float64{1, 1, 0, 0}
	outputTensor := tensor.New(tensor.WithBacking(outputVals), tensor.WithShape(4, 1))
	output := NewMatrix(g,
		tensor.Float64,
		WithName("y"),
		WithShape(4, 1),
		WithValue(outputTensor),
	)

	//Run forward pass
	if err := mind.forwardProp(input); err != nil {
		log.Fatalf("%+v", err)
	}

	//Calculate Cost w/MSE
	losses := Must(Sub(output, mind.outputNode))
	square := Must(Square(losses))
	cost := Must(Mean(square))

	//Do Gradient updates
	if _, err := Grad(cost, mind.learnables()...); err != nil {
		log.Fatal(err)
	}

	vm := NewTapeMachine(g, BindDualValues(mind.learnables()...))
	solver := NewVanillaSolver(WithLearnRate(0.1))
	million := 1000000
	trainingRounds := million / 10
	for i := 0; i < trainingRounds; i++ {
		vm.Reset()
		if err := vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(NodesToValueGrads(mind.learnables()))
		vm.Reset()
		if (i % 10000) == 0 {
			fmt.Printf("Iteration %d Completed\n", i)
		}
	}
	fmt.Println("\n\nOutput after Training: \n", mind.outputValue)
}

func AdditionNeuralNetworkTest() {
	nn := NewRawAI(2, 2, 1, 2/math.Pow(10, 14))
	err := nn.ImportWeights("AdditionAI.json")
	if err != nil {
		fmt.Print("No weights to import\n")
	}
	fmt.Printf("Weights IH Before: %v\n\nWeights HO After: %v\n", nn.WeightsIH, nn.WeightsHO)
	//Train Neural Network
	//
	for epoch := 0; epoch < 10000000; epoch++ {
		for i := 0; i <= 10; i++ {
			for j := 0; j <= 10; j++ {
				inputs := make([]float64, 2)
				targets := make([]float64, 1)
				inputs[0] = float64(i)
				inputs[1] = float64(j)
				targets[0] = float64(i) + float64(j)
				nn.Train(inputs, targets)
				if epoch%20000 == 0 && i == 5 && j == 5 {
					fmt.Printf("[TRAINING] [EPOCH %d] %f + %f = %f TARGETS[%f]\n", epoch, inputs[0], inputs[1], nn.OutputLayer[0], targets[0])
				}

			}

		}
	}
	// Test neural network
	a := rand.Intn(10) + 1
	b := rand.Intn(10) + 1
	inputs := make([]float64, 2)
	inputs[0] = float64(a)
	inputs[1] = float64(b)
	prediction := nn.FeedForward(inputs)[0]
	fmt.Printf("%d + %d = %f\n", a, b, prediction)
	fmt.Printf("Weights IH: %v\n\nWeights HO: %v\n", nn.WeightsIH, nn.WeightsHO)
	err = nn.ExportWeights("AdditionAI.json")
	if err != nil {
		panic(err)
	}
}

func UseAdditionNeuralNetwork() {
	nn := NewRawAI(2, 2, 1, 2/math.Pow(10, 14))
	err := nn.ImportWeights("AdditionAI.json")
	if err != nil {
		fmt.Print("No weights to import\n")
	}
	for a := 0; a < 20; a++ {
		for b := 0; b < 20; b++ {
			inputs := make([]float64, 2)
			inputs[0] = float64(a)
			inputs[1] = float64(b)
			prediction := nn.FeedForward(inputs)[0]
			fmt.Printf("%d + %d = %f\n", a, b, prediction)
		}
	}
}

func MultiplicationNeuralNetworkTest() {
	nn := NewRawAI(2, 2, 1, 1/math.Pow(10, 13))
	err := nn.ImportWeights("MultiplicationAI.json")
	if err != nil {
		fmt.Print("No weights to import\n")
	}
	fmt.Printf("Weights IH Before: %v\n\nWeights HO After: %v\n", nn.WeightsIH, nn.WeightsHO)
	//Train Neural Network
	//
	for epoch := 0; epoch < 10000000; epoch++ {
		for i := 0; i <= 10; i++ {
			for j := 0; j <= 10; j++ {
				inputs := make([]float64, 2)
				targets := make([]float64, 1)
				inputs[0] = float64(i)
				inputs[1] = float64(j)
				targets[0] = float64(i) * float64(j)
				nn.Train(inputs, targets)
				if epoch%20000 == 0 && i == 5 && j == 5 {
					fmt.Printf("[TRAINING] [EPOCH %d] %f * %f = %f TARGETS[%f]\n", epoch, inputs[0], inputs[1], nn.OutputLayer[0], targets[0])
				}

			}

		}
	}
	// Test neural network
	a := rand.Intn(10) + 1
	b := rand.Intn(10) + 1
	inputs := make([]float64, 2)
	inputs[0] = float64(a)
	inputs[1] = float64(b)
	prediction := nn.FeedForward(inputs)[0]
	fmt.Printf("%d + %d = %f\n", a, b, prediction)
	fmt.Printf("Weights IH: %v\n\nWeights HO: %v\n", nn.WeightsIH, nn.WeightsHO)
	err = nn.ExportWeights("MultiplicationAI.json")
	if err != nil {
		panic(err)
	}
}
