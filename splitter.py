
with open("spam.csv", "rb") as input_file, open("spam1.csv", "wb")  as spam_file , open("ham.csv", "wb")  as ham_file:
    for line in input_file:
        if line.startswith(b"spam"):
            spam_file.write(line)
        else:
            ham_file.write(line)
