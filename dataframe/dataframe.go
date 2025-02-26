package DataFrame

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
)

type DataFrame2D struct {
	Data [][]float64
}

func (frame *DataFrame2D) Shape() string {
	return fmt.Sprintf("(%dx%d)", len(frame.Data), len(frame.Data[0]))
}

func (frame *DataFrame2D) Plus(bias [][]float64) *DataFrame2D {
	result := make([][]float64, len(frame.Data))
	for i := range len(frame.Data) {
		result[i] = make([]float64, len(frame.Data[0]))
		for j := range len(frame.Data[0]) {
			result[i][j] = frame.Data[i][j] + bias[i][0]
		}
	}
	return &DataFrame2D{result}
}

func (frame *DataFrame2D) PrettyPrint() {
	for i := range frame.Data {
		fmt.Println(frame.Data[i])
	}
}

func (frame *DataFrame2D) Dot(that *DataFrame2D) (*DataFrame2D, error) {
	if frame.Data == nil {
		return nil, errors.New("empty data Frame")
	}
	if len(frame.Data[0]) != len(that.Data) {
		return nil, errors.New("matrix dimensions are not compatible")
	}

	m := len(frame.Data)
	q := len(that.Data[0])
	cols := len(frame.Data[0])

	resultData := make([][]float64, m)

	for i := range resultData {
		resultData[i] = make([]float64, q)
	}

	for i := range m {
		for j := range q {
			row := make([]float64, cols)
			copy(row, frame.Data[i])
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

func (frame *DataFrame2D) Transpose() {
	rows := len(frame.Data)
	cols := len(frame.Data[0])
	result := make([][]float64, cols)

	for i := range result {
		result[i] = make([]float64, rows)
	}

	for i := range cols {
		for j := range rows {
			result[i][j] = frame.Data[j][i]
		}
	}

	frame.Data = result
}

func (frame *DataFrame2D) ReadFromCsv(filePath string) {
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
	frame.Data = convertToIntMatrix(records[1:])
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
