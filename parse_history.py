import sys
import json
import re
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--len', default='7',type=int)
args = parser.parse_args()

def parse_tagging(sentences):
	taggings = []
	types = ['FROM-TO', 'REL', 'CAT']
	n = []
	length = []
	final_lines = ""
	pattern = '<[^\<\>]*>'
	WTF = ['%','% ','%uh', '%Uh', '%Uh ', '%um', '%um ', '%Um', '%Um ', '%oh ', '%Oh ','%hm ','%uh ','%hm ','%Hm','%Hm ','%Ah','%Ah ','%ah','%ah ','%eh','%eh ','%oh','%oh ','%h','%h ','%eh','%eh ','%Eh','%Eh ','%er','%er ','%Er','%Er ','%un','%un ']
	for line in sentences:
		a = re.findall(pattern, line)
		line = line.replace(',', '').replace('?', '').replace('.', '').replace('\r', '')
		for wtf in WTF:
			line = line.replace(wtf, '')
		tmp_line = line
		if (len(a) > 0):
			for s in a:
				tmp_line = tmp_line.replace(s, '')
		tmp_line = tmp_line.strip().replace('  ', ' ')
		if (tmp_line == ''):
			continue
		else:
			length.append(len(tmp_line.split(' ')))
		final_lines = tmp_line.lower().replace('-','')
		if ('<' in line):
			tmp = line.split('<')
			count = (int)((len(tmp) - 1) / 2)
			tmp_n = []
			tmp_taggings = []
			offset = 0
			for i in range(count):
				offset += len(tmp[i*2].split(' ')) - 1
				tmp_n.append(offset)
				tmp_str = tmp[i*2+1].split('>')[0].split(' ')
				tmp_tag = tmp_str[0]
				for j in range(len(types)):
					find = 0
					for k in range(len(tmp_str)):
						if (types[j] in tmp_str[k]):
							tmp_tag += '-' + tmp_str[k].split('\"')[1]
							find = 1
							break
					if (find == 0):
						tmp_tag += '-NONE'
				tmp_taggings.append('B-' + tmp_tag)
				tagwords_length = len(tmp[i*2+1].split('>')[1].strip().split(' '))
				for j in range(tagwords_length-1):
					tmp_n.append(offset+j+1)
					tmp_taggings.append('I-' + tmp_tag)
			taggings.append(tmp_taggings)
			n.append(tmp_n)
		else:
			taggings.append([])
			n.append([-1])

	final_tags = ""
	for i in range(len(taggings)):
		c = 0
		tmp_tags = ''
		for j in range(length[i]):
			if (j in n[i]):
				tmp_tags += taggings[i][c] + ' '
				c += 1
			else:
				tmp_tags += 'O '
		final_tags = tmp_tags

	return final_lines, final_tags

def parse_one_json(json_dir,speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,intype):
	with open(json_dir + '/label.json', 'r') as jsonfile:
		data = json.load(jsonfile)
	sentences = []
	counter = 0
	step_pref_guide = []
	step_pref_tourist = []
	nl_pref_guide = []
	nl_pref_tourist = []
	nl_pref = ""
	i_pref_guide = []
	i_pref_tourist = []
	i_pref = ""
	s_pref_guide = []
	s_pref_tourist = []
	s_pref = ""
	speaker = ""
	empty = "Empty"
	for i in range(args.len):
		step_pref_tourist.append(args.len-i)
		step_pref_guide.append(args.len-i)
		i_pref_tourist.append("None-none")
		s_pref_tourist.append('O')
		i_pref_guide.append("None-none")
		s_pref_guide.append('O')
		nl_pref_guide.append(empty)
		nl_pref_tourist.append(empty)
	for line in data["utterances"]:
		speaker_info = speaker_list[counter]
		counter += 1
		for i in range(len(line["semantic_tagged"])):
			"""
			SAP solo training, full information for SAP
			sentence = "guide_act-"+ speaker_info["guide_act"] + " "
			sentence += "initiativity-" + speaker_info["initiativity"] + " "
			sentence += "target_bio-" + speaker_info["target_bio"] + " "
			sentence += "topic-" + speaker_info["topic"] + " "
			sentence += "tourist_act-" + speaker_info["tourist_act"] + " "
			sentence += "speaker-" + speaker_info["speaker"] + " "
			"""
			sentence = line["semantic_tagged"][i].encode('utf-8')
			final_line, final_tag = parse_tagging([sentence])

			# sentence += line["speech_act"][i]["act"] + " "
			# for attr in line["speech_act"][i]["attributes"]:
			# 	if attr == "":
			# 		attr = "MAIN"
			# 	sentence += attr + " "
			intent = ""
			if line["speech_act"][i]["act"] == "":
				intent = "None"
			else:
				intent = line["speech_act"][i]["act"].strip()
			for attr in line["speech_act"][i]["attributes"]:
				attr = attr.strip()
				if attr == "":
					attr = 'none'
				intent += "-" + attr
			if final_line.strip() == "":
				nl_pref = "Empty"
			else:
				nl_pref = final_line
			i_pref = intent
			s_pref = final_tag
			speaker = speaker_info["speaker"].strip()
			if intype == "train":
				for s in nl_pref_tourist:
					f1.write(s + " ***next*** ")
				for s in nl_pref_guide:
					f1.write(s + " ***next*** ")
				f1.write(nl_pref + " ***next*** \n")

				for s in s_pref_tourist:
					f5.write(s + " ***next*** ")
				for s in s_pref_guide:
					f5.write(s + " ***next*** ")
				f5.write(s_pref + " ***next*** \n")

				for s in i_pref_tourist:
					f9.write(s + " ***next*** ")
				for s in i_pref_guide:
					f9.write(s + " ***next*** ")
				f9.write(i_pref + " ***next*** \n")

				for s in step_pref_tourist:
					f13.write(str(s) + " ***next*** ")
				for s in step_pref_guide:
					f13.write(str(s) + " ***next*** ")
				f13.write('0' + " ***next*** \n")
				f4.write(speaker+'\n')
			elif intype == "test":
				for s in nl_pref_tourist:
					f2.write(s + " ***next*** ")
				for s in nl_pref_guide:
					f2.write(s + " ***next*** ")
				f2.write(nl_pref + " ***next*** \n")

				for s in s_pref_tourist:
					f6.write(s + " ***next*** ")
				for s in s_pref_guide:
					f6.write(s + " ***next*** ")
				f6.write(s_pref + " ***next*** \n")

				for s in i_pref_tourist:
					f10.write(s + " ***next*** ")
				for s in i_pref_guide:
					f10.write(s + " ***next*** ")
				f10.write(i_pref + " ***next*** \n")

				for s in step_pref_tourist:
					f14.write(str(s) + " ***next*** ")
				for s in step_pref_guide:
					f14.write(str(s) + " ***next*** ")
				f14.write('0' + " ***next*** \n")
				f8.write(speaker+'\n')
			else:
				for s in nl_pref_tourist:
					f3.write(s + " ***next*** ")
				for s in nl_pref_guide:
					f3.write(s + " ***next*** ")
				f3.write(nl_pref + " ***next*** \n")
				for s in s_pref_tourist:
					f7.write(s + " ***next*** ")
				for s in s_pref_guide:
					f7.write(s + " ***next*** ")
				f7.write(s_pref + " ***next*** \n")
				for s in i_pref_tourist:
					f11.write(s + " ***next*** ")
				for s in i_pref_guide:
					f11.write(s + " ***next*** ")
				f11.write(i_pref + " ***next*** \n")
				for s in step_pref_tourist:
					f15.write(str(s) + " ***next*** ")
				for s in step_pref_guide:
					f15.write(str(s) + " ***next*** ")
				f15.write('0' + " ***next*** \n")
				f12.write(speaker+'\n')

				
			if speaker.strip() == "Tourist":
				nl_pref_tourist = nl_pref_tourist[1:]  
				i_pref_tourist = i_pref_tourist[1:]
				s_pref_tourist = s_pref_tourist[1:]
				step_pref_tourist = step_pref_tourist[1:]
				nl_pref_tourist.append(nl_pref)
				i_pref_tourist.append(i_pref)
				s_pref_tourist.append(s_pref)
				step_pref_tourist.append(0)
			else:
				nl_pref_guide = nl_pref_guide[1:]  
				i_pref_guide = i_pref_guide[1:]
				s_pref_guide = s_pref_guide[1:]
				step_pref_guide = step_pref_guide[1:]
				nl_pref_guide.append(nl_pref)
				i_pref_guide.append(i_pref)
				s_pref_guide.append(s_pref)
				step_pref_guide.append(0)
			for n in range(len(step_pref_tourist)):
				step_pref_tourist[n] = step_pref_tourist[n] + 1
				step_pref_guide[n] = step_pref_guide[n] + 1

def sent_2_speaker(json_dir):
	with open(json_dir + '/log.json', 'r') as jsonfile:
		data = json.load(jsonfile)
	speaker = []
	for line in data["utterances"]:
		info = dict()
		if "guide_act" in line["segment_info"]:
			info["guide_act"] = line["segment_info"]["guide_act"]
		else:
			info["guide_act"] = ""
		if "initiativity" in line["segment_info"]:
			info["initiativity"] = line["segment_info"]["initiativity"]
		else:
			info["initiativity"] = ""
		if "target_bio" in line["segment_info"]:
			info["target_bio"] = line["segment_info"]["target_bio"]
		else:
			info["target_bio"] = ""
		if "topic" in line["segment_info"]:
			info["topic"] = line["segment_info"]["topic"]
		else:
			info["topic"] = ""
		if "tourist_act" in line["segment_info"]:
			info["tourist_act"] = line["segment_info"]["tourist_act"]
		else:
			info["tourist_act"] = ""
		info["speaker"] = line["speaker"]
		speaker.append(info)
	return speaker

FinalLines = []
FinalTags = []
finalintent = []

f1 = open('./Data/train/seq.in','w')
f2 = open('./Data/test/seq.in','w')
f3 = open('./Data/valid/seq.in','w')
f4 = open('./Data/train/talker','w')
f5 = open('./Data/train/seq.out','w')
f6 = open('./Data/test/seq.out','w')
f7 = open('./Data/valid/seq.out','w')
f8 = open('./Data/test/talker','w')
f9 = open('./Data/train/intent','w')
f10 = open('./Data/test/intent','w')
f11 = open('./Data/valid/intent','w')
f12 = open('./Data/valid/talker','w')
f13 = open('./Data/train/info','w')
f14 = open('./Data/test/info','w')
f15 = open('./Data/valid/info','w')

#train_dir = ['001','002','003','004','006','007','008','009','010','011','012','013','016','017','019','020','021','022','023','024','025','026','028','030','031','032','033','035','039','040','041','047','048','052','053']
train_4 = ['001','002','003','004','006','007','008','009','010','012','013','017','019','022']
test_4 = ['021','023','024','030','033','035','041','047','048']
dev_4 =  ['011','016','020','025','026','028']
for i in range(len(test_4)):
	json_dir = './dstc4/'
	if (not os.path.exists( json_dir +test_4[i] +'/label.json')):
		print ('dir error')
	else:
		speaker_list = sent_2_speaker(json_dir + test_4[i])
		parse_one_json(json_dir + test_4[i],speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,"test")
for i in range(len(train_4)):
	json_dir = './dstc4/'
	if (not os.path.exists( json_dir +train_4[i] +'/label.json')):
		print ('dir error')
	else:
		speaker_list = sent_2_speaker(json_dir + train_4[i])
		parse_one_json(json_dir + train_4[i],speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,"train")
for i in range(len(dev_4)):
	json_dir = './dstc4/'
	if (not os.path.exists( json_dir +dev_4[i] +'/label.json')):
		print ('dir error')
	else:
		speaker_list = sent_2_speaker(json_dir + dev_4[i])
		parse_one_json(json_dir + dev_4[i],speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,"dev")
