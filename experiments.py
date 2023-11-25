import PlayChess

print("Select from the following board sizes for experiment")
print("1. 4x4")
print("2. 5x5")
print("3. 6x6")
print("4. 7x7")
print("5. 8x8")

size = int(input("Enter your choice"))
if(size == 1):
    PlayChess.experiment(4)
if(size == 2):
    PlayChess.experiment(5)
if(size == 3):
    PlayChess.experiment(6)
if(size == 4):
    PlayChess.experiment(7)
if(size == 5):
    PlayChess.experiment(8)
        
        