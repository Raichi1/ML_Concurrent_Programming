package factoreslatentes

import (
	"fmt"
	"sync"
	"time"
)

// Función para obtener recomendaciones concurrentemente
func getRecommendationsConcurrent(user string, ratings map[string]map[string]float64, similarityFunc func(map[string]float64, map[string]float64) float64) map[string]float64 {
	scores := make(map[string]float64)
	simSums := make(map[string]float64)

	// Normaliza las calificaciones del usuario objetivo
	normalizedUserRatings := normalizeRatings(ratings[user])

	var wg sync.WaitGroup
	var mu sync.Mutex

	// Goroutine para calcular similitud y recomendaciones
	for otherUser, otherRatings := range ratings {
		if otherUser == user {
			continue
		}

		wg.Add(1)
		go func(otherUser string, otherRatings map[string]float64) {
			defer wg.Done()

			// Normaliza las calificaciones del otro usuario
			normalizedOtherRatings := normalizeRatings(otherRatings)

			// Calcula la similitud (ya sea Pearson o Coseno)
			similarity := similarityFunc(normalizedUserRatings, normalizedOtherRatings)

			if similarity <= 0 {
				return
			}

			// Bloquear para acceder a scores y simSums concurrentemente
			mu.Lock()
			defer mu.Unlock()

			// Recorre los ítems calificados por otros usuarios
			for item, rating := range otherRatings {
				if _, ok := ratings[user][item]; !ok {
					// Suma ponderada de las calificaciones por la similitud
					scores[item] += similarity * rating
					simSums[item] += similarity
				}
			}
		}(otherUser, otherRatings)
	}

	// Espera a que todas las goroutines terminen
	wg.Wait()

	// Calcula la puntuación ponderada final
	for item := range scores {
		scores[item] /= simSums[item]
	}

	return scores
}

func UnderlyingFactorsConcurrent(ratings map[string]map[string]float64) {

	start := time.Now()

	user := "User100"

	// Escoge entre pearsonSimilarity o cosineSimilarity
	recommendations := getRecommendationsConcurrent(user, ratings, pearsonSimilarity)
	fmt.Println("Recomendaciones para", user, ":", recommendations)

	// También puedes probar con la similitud coseno:
	cosineRecommendations := getRecommendationsConcurrent(user, ratings, cosineSimilarity)
	fmt.Println("Recomendaciones con Coseno para", user, ":", cosineRecommendations)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
