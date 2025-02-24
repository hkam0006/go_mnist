package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"strconv"
)

type DataFrame2D struct {
	Data [][]float64
}

func (this *DataFrame2D) prettyPrint() {
	for i := range this.Data {
		fmt.Println(this.Data[i])
	}
}

func (this *DataFrame2D) dot(that *DataFrame2D) (*DataFrame2D, error) {
	if this.Data == nil {
		return nil, errors.New("Empty data frame")
	}
	if len(this.Data[0]) != len(that.Data) {
		return nil, errors.New("Matrix dimensions are not compatible")
	}

	m := len(this.Data)
	q := len(that.Data[0])
	cols := len(this.Data[0])

	resultData := make([][]float64, m)

	for i := range resultData {
		resultData[i] = make([]float64, q)
	}

	for i := range m {
		for j := range q {
			row := make([]float64, cols)
			copy(row, this.Data[i])
			product := 0.0
			for k := range cols {
				row[k] *= that.Data[k][j]
				product += row[k]
			}
			resultData[i][j] = product
		}
	}

	return &DataFrame2D{resultData}, nil
}

func (this *DataFrame2D) transpose() {
	rows := len(this.Data)
	cols := len(this.Data[0])
	result := make([][]float64, cols)

	for i := range result {
		result[i] = make([]float64, rows)
	}

	for i := range cols {
		for j := range rows {
			result[i][j] = this.Data[j][i]
		}
	}

	this.Data = result
}

func (this *DataFrame2D) readFromCsv(filePath string) {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()

	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	// data without headings, headings = records[0]
	this.Data = convertToIntMatrix(records[1:])
}

func (this *DataFrame2D) shape() string {
	return fmt.Sprintf("(%dx%d)", len(this.Data), len(this.Data[0]))
}

func convertToIntMatrix(records [][]string) [][]float64 {
	n := len(records)
	m := len(records[0])

	output := make([][]float64, n)

	for i := range output {
		output[i] = make([]float64, m)
	}

	for i := range n {
		for j := range m {
			integer, err := strconv.Atoi(records[i][j])
			if err != nil {
				log.Fatal("Unable to convert string to int")
			}
			floatNum := float64(integer)
			output[i][j] = floatNum
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
			result[i][j] = rand.Float64() - 0.5
		}
	}
	return result
}

// column headings = records[0]
// data = records[1:]
// label = record[0]
// pixels = record[1:] gray scale pixels
func main() {
	df := DataFrame2D{}
	df.readFromCsv("./dataset/mnist_train.csv")

	// Validation Data
	data_dev := DataFrame2D{
		Data: df.Data[0:1000],
	}

	data_dev.transpose()

	Y_dev := DataFrame2D{
		Data: append(make([][]float64, 0), data_dev.Data[0]),
	}

	X_dev := DataFrame2D{
		Data: data_dev.Data[1:],
	}

	fmt.Println("---------")
	fmt.Println("Val Label Shape: ", Y_dev.shape())
	fmt.Println("Val Data Shape: ", X_dev.shape())
	fmt.Println("---------")

	// Train Data
	train_data := DataFrame2D{
		Data: df.Data[1000:],
	}

	train_data.transpose()

	Y_train := DataFrame2D{
		Data: append(make([][]float64, 0), train_data.Data[0]),
	}

	X_train := DataFrame2D{
		Data: train_data.Data[1:],
	}

	fmt.Println("Val Label Shape: ", Y_train.shape())
	fmt.Println("Val Data Shape: ", X_train.shape())
	fmt.Println("---------")

	W1 := DataFrame2D{
		Data: generateRandom(10, 784),
	}

	W2 := DataFrame2D{
		Data: generateRandom(10, 10),
	}

	B1 := DataFrame2D{
		Data: generateRandom(10, 1),
	}

	B2 := DataFrame2D{
		Data: generateRandom(10, 1),
	}

	fmt.Println("W1 Shape: ", W1.shape())
	fmt.Println("B1 Shape: ", B1.shape())
	fmt.Println("W2 Shape: ", W2.shape())
	fmt.Println("B2 Shape: ", B2.shape())
	fmt.Println("---------")
}
