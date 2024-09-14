package svm

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// Estructura del modelo SVM
type SVMC struct {
	weights []float64
	bias    float64
	mu      sync.Mutex // Mutex para evitar condiciones de carrera
}

// Inicializa el modelo SVM
func newSVMConcurrent(inputSize int) *SVMC {
	weights := make([]float64, inputSize)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1 // Valores aleatorios entre -1 y 1
	}
	return &SVMC{
		weights: weights,
		bias:    rand.Float64()*2 - 1,
	}
}

// Calcula la predicción del modelo
func (svm *SVMC) predictConcurrent(inputs []float64) float64 {
	sum := svm.bias
	for i := range inputs {
		sum += inputs[i] * svm.weights[i]
	}
	if sum >= 0 {
		return 1.0
	}
	return 0.0
}

// Actualiza los pesos de forma concurrente
func (svm *SVMC) updateWeightsConcurrent(data []float64, label float64, prediction float64, learningRate float64, lambda float64) {
	svm.mu.Lock() // Bloquea el mutex para actualizar los pesos de forma segura
	defer svm.mu.Unlock()

	// Solo actualiza si el margen no se respeta
	if label*prediction < 1 {
		for j := range svm.weights {
			svm.weights[j] += learningRate * (label*data[j] - lambda*svm.weights[j])
		}
		svm.bias += learningRate * label
	} else {
		// Regularización
		for j := range svm.weights {
			svm.weights[j] -= learningRate * lambda * svm.weights[j]
		}
	}
}

// Entrena el modelo SVM concurrentemente usando goroutines
func (svm *SVMC) trainConcurrent(data [][]float64, labels []float64, epochs int, learningRate float64, lambda float64) {
	var wg sync.WaitGroup

	for epoch := 0; epoch < epochs; epoch++ {
		// Entrenar cada ejemplo de manera concurrente
		for i := range data {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				prediction := svm.predictConcurrent(data[idx])
				svm.updateWeightsConcurrent(data[idx], labels[idx], prediction, learningRate, lambda)
			}(i)
		}
		// Esperar a que todas las goroutines terminen
		wg.Wait()
	}
}

// Calcula la precisión
func accuracyConcurrent(predictions, labels []float64) float64 {
	correct := 0
	total := len(labels)
	for i := range predictions {
		if predictions[i] == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(total)
}

// Calcula la matriz de confusión
func confusionMatrixConcurrent(predictions, labels []float64) (int, int, int, int) {
	tp, tn, fp, fn := 0, 0, 0, 0
	for i := range predictions {
		if predictions[i] >= 0 && labels[i] == 1 {
			tp++
		} else if predictions[i] < 0 && labels[i] == 0 {
			tn++
		} else if predictions[i] >= 0 && labels[i] == 0 {
			fp++
		} else if predictions[i] < 0 && labels[i] == 1 {
			fn++
		}
	}
	return tp, tn, fp, fn
}

// Función principal que ejecuta el entrenamiento y evalúa el modelo
func SVMConcurrent(train [][]float64, label_train []float64, test [][]float64, label_test []float64) {
	// rand.Seed(42) // Inicializa la semilla aleatoria

	start := time.Now()

	// Inicializar el modelo SVM
	svm := newSVMConcurrent(len(train[0]))

	// Entrenar el modelo concurrentemente
	svm.trainConcurrent(train, label_train, 100, 0.01, 0.001)

	// Hacer predicciones en los datos de entrenamiento
	predictions := make([]float64, len(train))
	for i, d := range train {
		predictions[i] = svm.predictConcurrent(d)
	}

	// Evaluar el modelo
	acc := accuracyConcurrent(predictions, label_train)
	tp, _, fp, fn := confusionMatrixConcurrent(predictions, label_train)
	prec := precisionConcurrent(tp, fp)
	rec := recallConcurrent(tp, fn)
	f1 := f1ScoreConcurrent(prec, rec)

	fmt.Printf("Precision: %.2f\n", prec)
	fmt.Printf("Accuracy: %.2f\n", acc)
	fmt.Printf("F1-Score: %.2f\n", f1)
	fmt.Printf("Recall: %.2f\n", rec)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}

// Funciones adicionales para precisión, recall, y f1-score
func precisionConcurrent(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0.0
	}
	return float64(tp) / float64(tp+fp)
}

func recallConcurrent(tp, fn int) float64 {
	if tp+fn == 0 {
		return 0.0
	}
	return float64(tp) / float64(tp+fn)
}

func f1ScoreConcurrent(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * (precision * recall) / (precision + recall)
}
