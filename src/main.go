package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	recommendation "src/models/colaborative_filter" // Importar el paquete recommendation
	"strconv"
	"time"
)

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

func main() {

	fmt.Printf("=================================================\n")
	fmt.Printf("FILTRADO COLABORATIVO CONCURRENTE\n")
	colaborativeFilterCon()
	fmt.Printf("=================================================\n")
	fmt.Printf("FILTRADO COLABORATIVO SECUENCIAL\n")
	colaborativeFilter()
}
