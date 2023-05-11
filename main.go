package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type input struct {
	AuthorFollowers    int `json:"Author Followers"`
	AuthorLikes        int `json:"Author likes"`
	VideoLikesCount    int `json:"Video Likes Count"`
	VideoCommentsCount int `json:"Video Comments Count"`
	VideoSharesCount   int `json:"Video Shares Count"`
}

func readInput() (input input) {
	content, err := ioutil.ReadFile("input.json")
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(content, &input)
	if err != nil {
		panic(err)
	}
	return input
}
func main() {
	input := readInput()
	data := [][]string{}
	file, err := os.Open("data.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if err != nil {
		panic(err)
	}
	reader.Read() //SKIP ROW HEADER
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			panic(err)
		}
		data = append(data, row)
	}
	/*
		//Hiding Usernames
		for d := range data {
			data[d][0] = "Hidden"
		}
		f, _ := os.Create("hidden username data.csv")
		wr := csv.NewWriter(f)
		for _, d := range data {
			wr.Write(d)
		}
		f.Close()
	*/

	//Randomize
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	//Training
	XXTrain := [][]float64{}
	yTrain := []float64{}
	for _, row := range data {
		x := []float64{float64(toInt(row[1])), float64(toInt(row[2])), float64(toInt(row[5])), float64(toInt(row[6])), float64(toInt(row[7]))}
		minVal, maxVal := findMinAndMax(x)
		f := maxVal - minVal
		if f != 0 {
			XXTrain = append(XXTrain, normalize(x))
			yTrain = append(yTrain, float64(toInt(row[4])))
		}
	}

	weights := TRAIN(XXTrain, yTrain, 0.1, 1000)

	//Prediction

	prediction := dotProduct(normalize(
		[]float64{
			float64(input.AuthorFollowers),
			float64(input.AuthorLikes),
			float64(input.VideoLikesCount),
			float64(input.VideoCommentsCount),
			float64(input.VideoSharesCount),
		},
	), weights)
	fmt.Println("Video Count Predection:", int(prediction))

}
func findMinAndMax(a []float64) (min float64, max float64) {
	if len(a) <= 0 {
		return 0, 0
	}
	min = a[0]
	max = a[0]
	for _, value := range a {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}
	return min, max
}
func toInt(s string) int {
	i, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return i
}
func sum(arr []float64) float64 {
	s := 0.0
	for _, x := range arr {
		s += (x)
	}
	return s
}
func normalize(values []float64) []float64 {
	minVal, maxVal := findMinAndMax(values)
	f := maxVal - minVal
	normValues := make([]float64, len(values))
	for i, val := range values {
		normValues[i] = (val - minVal) / f
	}
	return normValues
}
func dotProduct(v1, v2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(v1); i++ {
		sum += v1[i] * v2[i]
	}
	return sum
}
func TRAIN(X [][]float64, y []float64, learningRate float64, epochs int) []float64 {
	weights := make([]float64, len(X[0]))

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(X); i++ {
			yPred := dotProduct(X[i], weights)
			Error := y[i] - yPred
			for j := 0; j < len(weights); j++ {
				weights[j] += learningRate * Error * X[i][j]
			}
		}
	}

	return weights
}
