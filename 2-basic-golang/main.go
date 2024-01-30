package main

import (
	"fmt"
	"sort"
)

// Main function
func main() {
	var amount int
	fmt.Scan(&amount)

	result := calculateChange(amount)
	printResult(result)
}

// Function to calculate the change and return it as a map
func calculateChange(amount int) map[int]int {
	fractions := [...]int{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}
	change := make(map[int]int)

	for i := len(fractions) - 1; i >= 0; i-- {
		if amount >= fractions[i] {
			result := amount / fractions[i]
			change[fractions[i]] = result
			amount = amount - result*fractions[i]
		}
	}

	return change
}

// Function to print calculation results
func printResult(m map[int]int) {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Sort(sort.Reverse(sort.IntSlice(keys)))

	for _, key := range keys {
		fmt.Printf("%d %d\n", key, m[key])
	}
}