from sklearn import datasets

houses_prices = datasets.fetch_california_housing()
digits = datasets.load_digits()

print(houses_prices.data)
print(digits.images[4])
