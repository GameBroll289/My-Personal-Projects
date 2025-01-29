import random

def rolls(Bet,M_G=10,P_G=1,M_S=5,P_S=1,M_B=3,P_B=1):
    options=[]
    slots=[]
    for i in range(P_G):
        options.append("G")
    for i in range(P_S):
        options.append("S")
    for i in range(P_B):
        options.append("B")
    for i in range(9):
        random_choice = random.choice(options)  # Randomly selects one element from the list
        slots.append(random_choice)
    display(slots)
    Earnings=logic(Bet,slots,M_G,M_S,M_B)
    return Earnings

def display(slots):
    print(f"\n{slots[0]}|{slots[1]}|{slots[2]}")
    print(f"{slots[3]}|{slots[4]}|{slots[5]}")
    print(f"{slots[6]}|{slots[7]}|{slots[8]}\n")

def logic(Bet,slots,M_G,M_S,M_B):
    Multi=0
    for i in range(0,9,3):
        if slots[i]==slots[i+1]==slots[i+2]:
            if slots[i]=="G":
                Multi+=M_G
            elif slots[i]=="S":
                Multi+=M_S
            elif slots[i]=="B":
                Multi+=M_B
    print(f"With your wagger being {Bet} and multiplier being {Multi}\nYou have earned: ",Bet*Multi)
    return Multi

def main():
    choice1=input("Enter Y if you want to enter the values of Slot Machine and if you want default values enter N: ").upper()
    if choice1=="Y":
        M_Gold=int(input("Set the multiplier of Gold: "))
        P_Gold=int(input("Set the probability of Gold: "))
        M_Silver=int(input("Set the multiplier of Silver: "))
        P_Silver=int(input("Set the probability of Silver: "))
        M_Bronze=int(input("Set the multiplier of Bronze: "))
        P_Bronze=int(input("Set the probability of Bronze: "))
        Balance=int(input("\nEnter your beginning deposit: "))
    elif choice1=="N":
        M_Gold,P_Gold,M_Silver,P_Silver,M_Bronze,P_Bronze=10,1,5,1,3,1
        Balance=int(300)
    while True:
        choice2=input(f"Your current balance is: {Balance}\nDo you want to continue? Enter Y if yes and N if no: ").upper()
        if choice2=="N":
            break
        elif choice2=="Y":
            Bet=int(input("\nHow much do you wanna wager: "))
        if Bet>Balance:
            print("\nInefficient Funds")
            break
        else:   
            Balance-=Bet
            Earnings=rolls(Bet,M_Gold,P_Gold,M_Silver,P_Silver,M_Bronze,P_Bronze)
            Balance+=Bet*Earnings


main()
print("Come again")
