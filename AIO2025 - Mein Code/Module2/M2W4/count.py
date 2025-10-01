class Solution:
    def calculate_variance_std(self, data):
        # calculate the mean
        mean = np.sum(data) / len(data)

        # calculate the variance
        variance = sum((x - mean) ** 2 for x in data) / len(data)

        # calculate the standard deviation
        std_dev = math.sqrt(variance)

        return variance, std_dev

