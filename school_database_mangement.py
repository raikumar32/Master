import sys


class School:

    def read_scores(self, file_name):

        self.fname = file_name

        # opening file

        f = open(file_name, 'r+')

        # reading first line

        line = f.readline()

        l = line.split()

        self.numstud = l[0][0]

        self.numcourse = l[0][1]

        # storing courses

        self.courses = l[1:]

        f.seek(0)

        # reading the first line

        f.readline()

        sid = []

        avg = []

        count = 0

        s = 0

        self.all_numbers = []

        for i in f:

            line = i.split()

            sid.append(line[0])

            numbers = line[1:]

            self.all_numbers.append(numbers)

            for j in numbers:

                if int(j) != -1 and int(j) != 888:
                    s = s + int(j)

                    count += 1

            avg.append(round(s / count))

            count = 0

            s = 0

        # LIST OF SSTUDENT OBJECTS

        std = []

        # creating list of student objects

        for i in sid:
            sob = Student(i)

            std.append(sob)

        # print(std)

        self.maxavg = 0

        self.topsid = ''

        # finding maximum avg

        for i in range(len(avg)):

            if avg[i] > self.maxavg:
                self.maxavg = avg[i]

                self.topsid = sid[i]

    # to display all information

    def disp(self):

        f = open(self.fname, 'r+')

        heading = f.readline()

        hd = heading.split()

        # print(hd)

        h = " | ".join(hd)

        print(" " * 5 + h[2:])

        print('------|' * (int(self.numcourse) + 1))

        for i in f:

            line = i.split()

            sid = line[0]

            print(sid + " ", end='|')

            nums = line[1:]

            for j in nums:

                if int(j) != -1 and int(j) != 888:

                    print("{0:^6}".format(j), end='|')
                elif int(j) == 888:
                    print("{0:^6}".format('--'), end='|')

                elif int(j) == -1:

                    print(' ' * 6, end='|')

            print()

        print(self.numstud, " students, ", end='')

        print(self.numcourse, " courses, ", end='')

        print('Top student is : ', self.topsid, end='')

        print(', average ', self.maxavg)

        f.close()


class Course:

    def __init__(self, cid=0, subj='', typ='', pt=0):

        self.__cid = cid

        self.__subject = subj

        self.__subj_type = typ

        self.__point = pt

    # setter methods

    def setCid(self, cid):

        self.__cid = cid

    def setSubject(self, sub):

        self.__subject = sub

    def setSubjectType(self, typ):

        if self.__subj_type.startswith("C") and typ.startswith('C'):
            self.__subj_type = typ

    def setPoint(self, pt):

        self.__point = pt

    # getter methods

    def getCid(self):

        return self.__cid

    def getSubject(self):

        return self.__subject

    def getSubjectType(self):

        return self.__subj_type

    def getPoint(self):

        return self.__point

    def disp(self, sch, fname):

        allnumbers = sch.all_numbers

        self.allcourses = sch.courses

        print(allnumbers)
        print(self.allcourses)

        total = 0

        index = 0

        count = 0

        self.course_info = {}

        col_length = len(allnumbers[0])

        row_length = len(allnumbers)

        self.avg_list = []

        for i in range(col_length):

            for j in range(row_length):

                # iterating over same column

                if int(allnumbers[j][i]) != -1 and int(allnumbers[j][i]) != 888:
                    # counting number of items

                    count += 1

                    # here key is the course name and

                    # no of items is the value

                    self.course_info[self.allcourses[index]] = count

                    # calculating total for same column(course)

                    total += int(allnumbers[j][i])

            # calculating average of this course

            # and storing in avg_list

            self.avg_list.append(round(total / count))

            count = 0

            total = 0

            index += 1

        # print(self.course_info)

        # print(self.avg_list)

        k = 0

        print("-" * 34)

        print('CID      Name         Pt. Enl. Avg.')

        print("-" * 34)

        f2 = open('courses_report.txt', 'w')

        f = open(fname)

        lowest = 99999999

        cid = ''

        f2.write('CID    Name         Pt. Enl. Avg.\n')

        f2.write("-" * 34 + '\n')

        for i in f:

            l = i.split()

            print(l[0], ' ', end='')

            f2.write(l[0])

            f2.write(' ')

            if l[2].startswith('C'):

                print('*', ' ', end='')

                f2.write('*')

                f2.write(' ')

            else:

                print('-', ' ', end='')

                f2.write('-')

                f2.write(' ')

            print(l[1].ljust(13), end='')

            f2.write(l[1].ljust(12))

            f2.write(' ')

            print(l[3].rjust(2), '  ', end='')

            f2.write(str(l[3].rjust(2)))

            f2.write('   ')

            print(self.course_info[l[0]], ' ', end='')

            f2.write(str(self.course_info[l[0]]))

            f2.write('   ')

            print(" " + str(self.avg_list[k]))

            f2.write(" " + str(self.avg_list[k]))

            f2.write('    ')

            if self.avg_list[k] < lowest:
                lowest = self.avg_list[k]

                cid = l[0]

            k += 1

            f2.write("\n")

        print()

        f2.write("-" * 34 + '\n')

        f2.write('The worse performing course is' + ' ' + str(cid) + ' ' + 'with an average' + ' ' + str(lowest))

        f2.close()

        print("-" * 34)

        print('The worse performing course is', cid, 'with an average', lowest)
        print()
        print("courses_report.txt generated! ")

        # print(self.course_info)


class Student:

    def __init__(self, s):
        self.SID = s

    def __str__(self):
        return 'S' + str(self.SID)


s1 = School()

fname = ''

try:

    fname = sys.argv[1]

    s1.read_scores(fname)

    s1.disp()

except:

    print(" No file is supplied")

c1 = Course()

c1.disp(s1, sys.argv[2])
