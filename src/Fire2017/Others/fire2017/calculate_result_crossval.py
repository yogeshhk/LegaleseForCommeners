
results_file = "results_crossval.data"
keywords_g = []
keywords_p = []
current_word_g = ""
current_word_p = ""
num_mismatches = 0
total_lines = 0
with open(results_file, 'r') as infile:
	for line in infile:
		words = line.split()
		if len(words) != 4:
			continue
		total_lines = total_lines + 1
		word, _,tag_g, tag_p = line.split()
		if tag_g != tag_p:
			num_mismatches = num_mismatches + 1
		if tag_g == "B-LEGAL":
			current_word_g = word
		elif tag_g == "I-LEGAL":
			current_word_g = current_word_g + " " + word
		elif current_word_g != "":
			keywords_g.append(current_word_g)
			current_word_g = ""

		if tag_p == "B-LEGAL":
			current_word_p = word
		elif tag_p == "I-LEGAL":
			current_word_p = current_word_p + " " + word
		elif current_word_p != "":
			keywords_p.append(current_word_p)
			current_word_p = ""

		gold_set = set(keywords_g)
		predict_set = set(keywords_p)
		intersection_set = gold_set.intersection(predict_set)
print("Gold words {}".format(len(gold_set)))
print("Predicted words {}".format(len(predict_set)))
print("Gold words {}".format(list(gold_set)))
print("Predicted words {}".format(list(predict_set)))
print("Common words {} ".format(len(intersection_set)))
print("Common words to total words ratio {}".format(len(intersection_set)/len(gold_set)))
print("Num Mismatches {} out of {}".format(num_mismatches, total_lines))