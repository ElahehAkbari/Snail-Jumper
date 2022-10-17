import matplotlib.pylab as plt

max = list()
min = list()
avg = list()

with open("result.txt", "r") as file:
    lines = file.readlines()

for i in lines:
    numbers = i.split(" ")
    max.append(float(numbers[0]))
    avg.append(float(numbers[1]))
    min.append(float(numbers[2]))

plt.plot(min, color='g', label="Min")
plt.plot(avg, color='r', label="Average")
plt.plot(max, color='b', label="Max")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()