# for remove line breaks in NOTEEVENTS.csv file
readin  = open('NOTEEVENTS.csv', 'r')
writeout = open('NOTEEVENTS_PROCESSED.csv','w')

header = readin.readline()
writeout.write(header)

quotes = 0
count = 0
while True:
	line = readin.readline()
	if not line:
		break
	new_line = ' '.join(line.replace("\n"," NEWLINE ").split())
	quotes += line.count("\"")
	while quotes % 2 == 1:
		line = readin.readline()
		quotes += line.count("\"")
		new_line += ' '
		if quotes % 2 == 1:
			new_line += ' '.join(line.replace("\n"," NEWLINE ").split())
		else:
			new_line += ' '.join(line.replace("\n"," ").split())
	new_line += "\n"
	count+=1
	writeout.write(new_line)
