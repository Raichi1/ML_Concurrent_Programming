package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	split "src/data"
	ann "src/models/ann"
	recommendation "src/models/colaborative_filter"
	decisiontree "src/models/decision_tree"
	dnn "src/models/dnn"
	randomforest "src/models/random_forest"
	svmachine "src/models/svm"
	"strconv"
	"time"
)

func splitFeaturesAndTarget(data [][]float64) ([][]float64, []float64) {
	numRows := len(data)
	numCols := len(data[0])

	features := make([][]float64, numRows)
	target := make([]float64, numRows)

	for i := 0; i < numRows; i++ {
		target[i] = data[i][numCols-1]
		features[i] = data[i][:numCols-1]
	}

	return features, target
}

func readCSV(filePath string, skipHeader bool) ([][]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Si el CSV tiene encabezado, lo eliminamos si skipHeader es true
	if skipHeader {
		records = records[1:]
	}

	return records, nil
}

func convertToInt(lista []float64) []int {
	intList := make([]int, len(lista))

	for i, f := range lista {
		intList[i] = int(f)
	}

	return intList
}

func convertToFloat(records [][]string) ([][]float64, error) {
	floatRecords := make([][]float64, len(records))
	for i, row := range records {
		floatRow := make([]float64, len(row))
		for j, value := range row {
			floatValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			floatRow[j] = floatValue
		}
		floatRecords[i] = floatRow
	}
	return floatRecords, nil
}

func getDataFrame(filepath string, skipHeader bool) ([][]float64, []float64, error) {
	records, err := readCSV(filepath, skipHeader)
	if err != nil {
		return nil, nil, err
	}

	data, err := convertToFloat(records)
	if err != nil {
		return nil, nil, err
	}

	features, target := splitFeaturesAndTarget(data)
	return features, target, nil
}

func colaborativeFilterCon() {
	// Abrir el archivo CSV
	file, err := os.Open("dataset/clean_songs.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Crear un lector CSV
	reader := csv.NewReader(file)

	// Leer todas las filas del archivo
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// La primera fila contiene los nombres de las columnas (ítems)
	columns := records[0]

	// tiempo de inicio
	start := time.Now()

	// Convertir las filas restantes en una matriz de calificaciones (ratings)
	var ratings [][]float64
	for _, row := range records[1:] {
		var userRatings []float64
		for _, val := range row {
			rating, err := strconv.ParseFloat(val, 64)
			if err != nil {
				userRatings = append(userRatings, 0) // Si hay un error, asignar 0
			} else {
				userRatings = append(userRatings, rating)
			}
		}
		ratings = append(ratings, userRatings)
	}

	// Índice del usuario para el cual queremos hacer recomendaciones
	userIndex := 1

	// Índices de los ítems para los cuales queremos predecir las calificaciones
	itemIndices := []int{2, 3} // Puedes cambiar esto según los ítems que te interesen

	// Obtener recomendaciones llamando a la función en el paquete recommendation
	recommendations := recommendation.GetRecommendationsC(ratings, userIndex, itemIndices)

	// Mostrar las recomendaciones
	for i, rating := range recommendations {
		fmt.Printf("Predicted rating for item '%s': %.2f\n", columns[itemIndices[i]], rating)
	}

	// calculo de tiempo
	elapsed := time.Since(start)

	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}

func colaborativeFilter() {
	// Abrir el archivo CSV
	file, err := os.Open("dataset/clean_songs.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Crear un lector CSV
	reader := csv.NewReader(file)

	// Leer todas las filas del archivo
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// La primera fila contiene los nombres de las columnas (ítems)
	columns := records[0]

	// tiempo de inicio
	start := time.Now()

	// Convertir las filas restantes en una matriz de calificaciones (ratings)
	var ratings [][]float64
	for _, row := range records[1:] {
		var userRatings []float64
		for _, val := range row {
			rating, err := strconv.ParseFloat(val, 64)
			if err != nil {
				userRatings = append(userRatings, 0) // Si hay un error, asignar 0
			} else {
				userRatings = append(userRatings, rating)
			}
		}
		ratings = append(ratings, userRatings)
	}

	// Índice del usuario para el cual queremos hacer recomendaciones
	userIndex := 1

	// Índices de los ítems para los cuales queremos predecir las calificaciones
	itemIndices := []int{2, 3} // Puedes cambiar esto según los ítems que te interesen

	// Obtener recomendaciones llamando a la función en el paquete recommendation
	recommendations := recommendation.GetRecommendations(ratings, userIndex, itemIndices)

	// Mostrar las recomendaciones
	for i, rating := range recommendations {
		fmt.Printf("Predicted rating for item '%s': %.2f\n", columns[itemIndices[i]], rating)
	}

	// calculo de tiempo
	elapsed := time.Since(start)

	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}

func svmSecuential(filepath string) {
	features, target, _ := getDataFrame(filepath, true)

	train, test, labelTrain, labelTest, err := split.SplitData(features, target, 0.2)

	if err != nil {
		log.Fatal(err)
		return
	}

	svmachine.SVMSecuential(train, labelTrain, test, labelTest)

}

func svmConcurrent(filepath string) {
	features, target, _ := getDataFrame(filepath, true)

	train, test, labelTrain, labelTest, err := split.SplitData(features, target, 0.2)

	if err != nil {
		log.Fatal(err)
		return
	}

	svmachine.SVMConcurrent(train, labelTrain, test, labelTest)
}

func decisionTreeSecuencial(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	decisiontree.DecisionTreeSec(features, labels)

}

func decisionTreeConcurrent(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	decisiontree.DecisionTreeConcurrente(features, labels)
}

func annSecuential(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	ann.ANNSecuential(features, labels)
}

func annConcurrent(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	ann.ANNConcurrent(features, labels)
}

func rfSecuential(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	train, test, trainLabel, testLabel, _ := split.SplitData(features, labels, 0.2)

	trainL := convertToInt(trainLabel)
	testL := convertToInt(testLabel)

	randomforest.RandomForestSecuential(train, trainL, test, testL)
}

func rfConcurrent(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	train, test, trainLabel, testLabel, _ := split.SplitData(features, labels, 0.2)

	trainL := convertToInt(trainLabel)
	testL := convertToInt(testLabel)

	randomforest.RandomForestSecuential(train, trainL, test, testL)
}

func dnnSecuential(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	train, test, trainLabel, testLabel, _ := split.SplitData(features, labels, 0.2)
	dnn.DNNSecuential(train, trainLabel, test, testLabel)
}

func dnnConcurrent(filepath string) {
	features, labels, _ := getDataFrame(filepath, true)

	train, test, trainLabel, testLabel, _ := split.SplitData(features, labels, 0.2)
	dnn.DNNConcurrent(train, trainLabel, test, testLabel)
}

func main() {

	filepath := "dataset/bank.csv"

	fmt.Printf("=================================================\n")
	fmt.Printf("FILTRADO COLABORATIVO CONCURRENTE\n")
	colaborativeFilterCon()
	fmt.Printf("=================================================\n")
	fmt.Printf("FILTRADO COLABORATIVO SECUENCIAL\n")
	colaborativeFilter()
	fmt.Printf("=================================================\n")
	fmt.Printf("SUPPORT VECTORIAL MACHINE CONCURRENTE\n")
	svmConcurrent(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("SUPPORT VECTORIAL MACHINE SECUENCIAL\n")
	svmSecuential(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("DECISION TREE SECUENCIAL\n")
	decisionTreeSecuencial(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("DECISION TREE CONCURRENTE\n")
	decisionTreeConcurrent(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("ARTIFICIAL NEURONAL NETWORK SECUENTIAL\n")
	annSecuential(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("ARTIFICIAL NEURONAL NETWORK CONCURRENTE\n")
	annConcurrent(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("RANDOM FOREST SECUENTIAL\n")
	rfSecuential(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("RANDOM FOREST CONCURRENT\n")
	rfSecuential(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("DEEP NEURONAL NETWORK SECUENTIAL\n")
	dnnSecuential(filepath)
	fmt.Printf("=================================================\n")
	fmt.Printf("DEEP NEURONAL NETWORK CONCURRENT\n")
	dnnConcurrent(filepath)
}
