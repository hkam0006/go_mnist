package main

import (
	"fmt"
	Frame "hkam0006/dataframe"
	"math"
	"math/rand"
)

type ActivationFunction interface {
	Pass([][]float64) [][]float64
}

func roundTwoFP(num float64) float64 {
	return math.Floor(num*100) / 100
}

type ReLu struct{}

func (this ReLu) Pass(arr [][]float64) [][]float64 {
	output := make([][]float64, len(arr))
	for i := range len(arr) {
		output[i] = make([]float64, len(arr[0]))
		for j := range len(arr[0]) {
			output[i][j] = max(arr[i][j], 0.0)
		}
	}
	return output
}

func generateRandom(n, m int) [][]float64 {
	result := make([][]float64, n)
	for i := range result {
		result[i] = make([]float64, m)
	}
	for i := range n {
		for j := range m {
			randomFloat := rand.Float64() - 0.5
			result[i][j] = roundTwoFP(randomFloat)
		}
	}
	return result
}

func softMax(d *Frame.DataFrame2D) *Frame.DataFrame2D {
	Z := make([][]float64, 10)
	sum_cols := make([]float64, len(d.Data[0]))
	for i := range len(d.Data) {
		for j := range len(d.Data[0]) {
			exponent := math.Exp(roundTwoFP(d.Data[i][j]))
			sum_cols[j] += exponent
		}
	}

	for i := range len(d.Data) {
		Z[i] = make([]float64, len(d.Data[0]))
		for j := range len(d.Data[0]) {
			numerator := math.Exp(roundTwoFP(d.Data[i][j]))
			Z[i][j] = roundTwoFP(numerator) / roundTwoFP(sum_cols[i])
			// if Z[i][j] == math.Inf(1) {
			// 	fmt.Println(Z[i][j])
			// }
		}
	}

	return &Frame.DataFrame2D{
		Data: Z,
	}
}

func forward_prop(W1, B1, W2, B2, X *Frame.DataFrame2D, a ActivationFunction) *Frame.DataFrame2D {
	Z1, err := W1.Dot(X)

	if err != nil {
		fmt.Println(err)
		return nil
	}

	Z1 = Z1.Plus(B1.Data)

	A1 := &Frame.DataFrame2D{
		Data: a.Pass(Z1.Data),
	}

	Z2, err := W2.Dot(A1)
	if err != nil {
		fmt.Println(err)
		return nil
	}
	Z2 = Z2.Plus(B2.Data)

	return softMax(Z2)
}

// column headings = records[0]
// data = records[1:]
// label = record[0]
// pixels = record[1:] gray scale pixels
func main() {
	df := Frame.DataFrame2D{}
	df.ReadFromCsv("./dataset/mnist_train.csv")

	// Validation Data
	data_dev := Frame.DataFrame2D{
		Data: df.Data[0:1000],
	}

	data_dev.Transpose()

	Y_dev := Frame.DataFrame2D{
		Data: append(make([][]float64, 0), data_dev.Data[0]),
	}

	X_dev := Frame.DataFrame2D{
		Data: data_dev.Data[1:],
	}

	fmt.Println("---------")
	fmt.Println("Val Label Shape: ", Y_dev.Shape())
	fmt.Println("Val Data Shape: ", X_dev.Shape())
	fmt.Println("---------")

	// Train Data
	train_data := Frame.DataFrame2D{
		Data: df.Data[1000:],
	}

	train_data.Transpose()

	Y_train := Frame.DataFrame2D{
		Data: append(make([][]float64, 0), train_data.Data[0]),
	}

	X_train := Frame.DataFrame2D{
		Data: train_data.Data[1:],
	}

	fmt.Println("Train Label Shape: ", Y_train.Shape())
	fmt.Println("Train Data Shape: ", X_train.Shape())
	fmt.Println("---------")

	W1 := Frame.DataFrame2D{
		Data: generateRandom(10, 784),
	}

	W2 := Frame.DataFrame2D{
		Data: generateRandom(10, 10),
	}

	B1 := Frame.DataFrame2D{
		Data: generateRandom(10, 1),
	}

	B2 := Frame.DataFrame2D{
		Data: generateRandom(10, 1),
	}

	fmt.Println("W1 Shape: ", W1.Shape())
	fmt.Println("B1 Shape: ", B1.Shape())
	fmt.Println("W2 Shape: ", W2.Shape())
	fmt.Println("B2 Shape: ", B2.Shape())
	fmt.Println("---------")

	forward_prop(
		&W1,
		&B1,
		&W2,
		&B2,
		&X_train,
		ReLu{},
	)
}
