package factoreslatentes

import (
	"fmt"
	"math"
	"time"
)

// Datos de ejemplo: usuarios y sus calificaciones de películas
// var ratings = map[string]map[string]float64{
// 	"User1": {"Movie1": 5, "Movie2": 3, "Movie3": 4},
// 	"User2": {"Movie1": 4, "Movie2": 2, "Movie3": 5, "Movie4": 1},
// 	"User3": {"Movie1": 1, "Movie2": 5, "Movie3": 3, "Movie4": 4, "Movie5": 5},
// 	"User4": {"Movie2": 4, "Movie3": 4, "Movie4": 5},
// 	"User5": {"Movie1": 3, "Movie2": 3, "Movie3": 4, "Movie5": 5},
// }

// Función para normalizar las calificaciones del usuario restando la media
func normalizeRatings(userRatings map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	var sum, count float64

	for _, rating := range userRatings {
		sum += rating
		count++
	}

	// Media de las calificaciones
	mean := sum / count

	for item, rating := range userRatings {
		normalized[item] = rating - mean
	}

	return normalized
}

// Función para calcular la similitud usando la correlación de Pearson
func pearsonSimilarity(user1, user2 map[string]float64) float64 {
	var sum1, sum2, sum1Sq, sum2Sq, pSum float64
	var n int

	for item := range user1 {
		if _, ok := user2[item]; ok {
			n++
			sum1 += user1[item]
			sum2 += user2[item]
			sum1Sq += math.Pow(user1[item], 2)
			sum2Sq += math.Pow(user2[item], 2)
			pSum += user1[item] * user2[item]
		}
	}

	if n == 0 {
		return 0
	}

	num := pSum - (sum1 * sum2 / float64(n))
	den := math.Sqrt((sum1Sq - math.Pow(sum1, 2)/float64(n)) * (sum2Sq - math.Pow(sum2, 2)/float64(n)))

	if den == 0 {
		return 0
	}

	return num / den
}

// Función para calcular la similitud coseno
func cosineSimilarity(user1, user2 map[string]float64) float64 {
	var sumXY, sumX, sumY float64

	for item := range user1 {
		if _, ok := user2[item]; ok {
			sumXY += user1[item] * user2[item]
			sumX += math.Pow(user1[item], 2)
			sumY += math.Pow(user2[item], 2)
		}
	}

	if sumX == 0 || sumY == 0 {
		return 0
	}

	return sumXY / (math.Sqrt(sumX) * math.Sqrt(sumY))
}

// Función para obtener recomendaciones
func getRecommendations(user string, ratings map[string]map[string]float64, similarityFunc func(map[string]float64, map[string]float64) float64) map[string]float64 {
	scores := make(map[string]float64)
	simSums := make(map[string]float64)

	// Normaliza las calificaciones del usuario objetivo
	normalizedUserRatings := normalizeRatings(ratings[user])

	for otherUser, otherRatings := range ratings {
		if otherUser == user {
			continue
		}

		// Normaliza las calificaciones del otro usuario
		normalizedOtherRatings := normalizeRatings(otherRatings)

		// Calcula la similitud (ya sea Pearson o Coseno)
		similarity := similarityFunc(normalizedUserRatings, normalizedOtherRatings)

		if similarity <= 0 {
			continue
		}

		// fmt.Printf("Similitud entre %s y %s: %f\n", user, otherUser, similarity)

		// Recorre los ítems calificados por otros usuarios
		for item, rating := range otherRatings {
			if _, ok := ratings[user][item]; !ok {
				// Suma ponderada de las calificaciones por la similitud
				scores[item] += similarity * rating
				simSums[item] += similarity
			}
		}
	}

	// Calcula la puntuación ponderada final
	for item := range scores {
		scores[item] /= simSums[item]
	}

	return scores
}

func UnderlyingFactors(ratings map[string]map[string]float64) {

	start := time.Now()

	user := "User100"

	// Escoge entre pearsonSimilarity o cosineSimilarity
	recommendations := getRecommendations(user, ratings, pearsonSimilarity)
	fmt.Println("Recomendaciones para", user, ":", recommendations)

	// También puedes probar con la similitud coseno:
	cosineRecommendations := getRecommendations(user, ratings, cosineSimilarity)
	fmt.Println("Recomendaciones con Coseno para", user, ":", cosineRecommendations)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s", elapsed)
}
