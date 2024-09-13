package main

import (
	"math"
	"sync"
)

var ratings = map[string]map[string]float64{
	"User1": {"Item1": 5, "Item2": 3, "Item3": 4},
	"User2": {"Item1": 4, "Item2": 2, "Item3": 5},
	"User3": {"Item1": 1, "Item2": 5, "Item3": 3},
}

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

func getRecommendationsConcurrent(user string, ratings map[string]map[string]float64) map[string]float64 {
	scores := make(map[string]float64)
	simSums := make(map[string]float64)
	var mu sync.Mutex

	var wg sync.WaitGroup

	for otherUser, otherRatings := range ratings {
		if otherUser == user {
			continue
		}

		wg.Add(1)
		go func(otherUser string, otherRatings map[string]float64) {
			defer wg.Done()
			similarity := pearsonSimilarity(ratings[user], otherRatings)

			if similarity <= 0 {
				return
			}

			mu.Lock()
			defer mu.Unlock()

			for item, rating := range otherRatings {
				if _, ok := ratings[user][item]; !ok {
					scores[item] += similarity * rating
					simSums[item] += similarity
				}
			}
		}(otherUser, otherRatings)
	}

	wg.Wait()

	for item := range scores {
		scores[item] /= simSums[item]
	}

	return scores
}

// func main() {
// 	user := "User1"
// 	recommendations := getRecommendationsConcurrent(user, ratings)
// 	fmt.Println("Recomendaciones concurrentes para", user, ":", recommendations)
// }
