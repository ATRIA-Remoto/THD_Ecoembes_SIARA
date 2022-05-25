import os, time

PATH = "./backup_train"

try:
    while True:
        listFiles = os.listdir(PATH)

        for f in listFiles:
            if "_last" in f:
                time.sleep(5)
                numSteps = 0
                for f2 in listFiles:
                    if "Steps" in f2:
                        index1 = f2.find("_")
                        index2 = index1 + 1 + f2[(index1+1):].find("_")
                        f2Steps = int(f2[(index1+1):index2])
                        if f2Steps > numSteps:
                            numSteps = f2Steps
                currentSteps = numSteps + 100
                newFileName = f[:f.find("_")] + "_" + str(currentSteps) + "_Steps.weights"
                os.rename(os.path.join(PATH, f), os.path.join(PATH, newFileName))
        
        time.sleep(1000)
except KeyboardInterrupt:
    print("Program finished")

