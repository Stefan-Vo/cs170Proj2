import random
import itertools

def stubEval(feature):
    num = random.randrange(0, 101, 2)
    return num

def forwardElimination(num):
    print("Forward Elimination")
    features = [{i} for i in range(1, num + 1)]
    print(features)
    size = len(features)
    origSize = size

    full_range = set(range(1, size + 1))
    
    bestAcc = 0
    bestFeature = set()
    
    while size > 0:
        change = False
        newFeatures = []
        for feature in features:
            currentAcc = stubEval(feature)
            print(f"        Using feature(s) {feature} accuracy is {currentAcc}%")
            if currentAcc >= bestAcc:
                change = True
                bestAcc = currentAcc
                bestFeature = feature
        print("")
        if change == True:
            print(f"Feature set {bestFeature} was best, accuracy is {bestAcc}%\n")
        else:
            print("(Warning, Accuracy has decreased!)")
            features = bestFeature
            break
        if len(bestFeature) == origSize:
            features = bestFeature
            break
        missing = full_range - bestFeature
        for number in missing:
            new_test = bestFeature.copy()
            new_test.add(number)
            newFeatures.append(new_test)
        features = newFeatures
        size -= 1
    print(f"Finished search!! The best feature subset is {features}, which has an accuracy of {bestAcc}%")
    return 0

def backwardsElimination(num):
    print("Backward Elimination")
    features = set()
    for i in range (1, num + 1):
        features.add(i)
    features = [features]
    print(features)
    origSize = num
    bestAcc = 0
    bestFeature = set()
    while num > 0:
        change = False
        for feature in features:
            currentAcc = stubEval(feature)
            print(f"        Using feature(s) {feature} accuracy is {currentAcc}%")
            if currentAcc > bestAcc:
                change = True
                bestAcc = currentAcc
                bestFeature = feature
        print("")
        if change == True:
            print(f"Feature set {bestFeature} was best, accuracy is {bestAcc}%\n")
        else:
            print("(Warning, Accuracy has decreased!)")
            features = bestFeature
            break
        if len(bestFeature) == 1:
            features = bestFeature
            break
        features = [set(combo) for combo in itertools.combinations(bestFeature, len(bestFeature) - 1)]
        num -= 1
        print(features)
    print(features)
    return 0

def main():
    print("Welcome to Daniel's Feature Selection Algorithm\n")

    num = int(input("Please enter the total number of features: "))

    print("\nType the number of the algorithm you want to run.")
    print("1: Forward Selection")
    print("2: Backward Elimination")

    alg = input()

    if alg == '1':
        forwardElimination(num)
    elif alg == '2':
        backwardsElimination(num)

# Call the main function
main()