package main

import (
	"fmt"
	"log"

	. "gorgonia.org/gorgonia"
)

// Basic example of representing mathematical equations as graphs.
//
// In this example, we want to represent the following equation
//		z = x + y
func Example_basic() {
	g := NewGraph()

	var x, y, z, w *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}
	if w, err = Mul(x, y); err != nil {
		log.Fatal(err)
	}
	// create a VM to run the program on
	machine := NewTapeMachine(g)
	defer machine.Close()

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Z=%v\n", z.Value())
	fmt.Printf("W=%v\n", w.Value())
	// Output: 4.5

}

func main() {
	//XORNeuralNetworkTest()
	//AdditionNeuralNetworkTest()
	//UseAdditionNeuralNetwork()
	MultiplicationNeuralNetworkTest()
}
