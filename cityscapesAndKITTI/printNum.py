def allnumberCandivide(num,divide):
    for i in range(1,divide+1):
        if num%i==0:
            print(i)

if __name__ == '__main__':
    num = int(input("Please input a number:"))
    print(allnumberCandivide(num, num))