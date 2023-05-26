import re
# import nltk
# nltk.download('punkt') 

"""
Meta Characters
    [] A set of characters
    \ Signals a special sequence (can also be used to escape special characters)
    . Any character (except newline character)
    ^ Starts with
    $ Ends with
    * Zero or more occurrences
    + One or more occurrences
    {} Exactly the specified number of occurrences
    | Either or
    () Capture and group

Special Sequences
    \A Returns a match if the specified characters are at the beginning of the string
    \b Returns a match where the specified characters are at the beginning or at the end of a word r"ain\b"
    \B Returns a match where the specified characters are present, but NOT at the beginning (or at the end) of a word

    \d Returns a match where the string contains digits (numbers from 0-9)
    \D Returns a match where the string DOES NOT contain digits
    \s Returns a match where the string contains a white space character
    \S Returns a match where the string DOES NOT contain a white space character
    \w Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, and the underscore _ character)
    \W Returns a match where the string DOES NOT contain any word characters
    \Z Returns a match if the specified characters are at the end of the string
"""


def clean(s):
    """
    Objective:
    1. <HASHTAG>
    2. <EMAIL>
    3. <MENTION>
    4. <NUMBER>
    5. <URL>
    6. <TIME>
    7. <MONEY>
    """

    # The following lines clean the text by inserting placeholders
    # and removing repetition.
    

    # Removing . just after a capital letter
    s = re.sub(r'Mrs.', 'Mrs', s)
    s = re.sub(r'Mr.', 'Mr', s)
    s = re.sub(r'Dr.', 'Dr', s)
    s = re.sub(r'i.e.', 'ie', s)
    s = re.sub(r'St.', 'St', s)
    s = re.sub(r'Jr.', 'Jr', s)
    s = re.sub(r'Prof.', 'Prof', s)
    s = re.sub(r'Rev.', 'Rev', s)
    s = re.sub(r'([A-Z])\.', r'\1', s)



    # can't to can not
    s = re.sub(r'([a-zA-Z]+)n\'t', r'\1 not', s)
    # Replacing i'm.
    s = re.sub(r'([iI])\'m', r'\1 am', s)
    # Replacing we've, i've.
    s = re.sub(r'([a-zA-Z]+)\'ve', r'\1 have', s)
    # Replacing i'd, they'd.
    s = re.sub(r'([a-zA-Z]+)\'d', r'\1 had', s)
    # Replacing i'll, they'll.
    s = re.sub(r'([a-zA-Z]+)\'ll', r'\1 will', s)
    # Replacing we're, they're.
    s = re.sub(r'([a-zA-Z]+)\'re', r'\1 are', s)
    # Replacing tryin', doin'.
    s = re.sub(r'([a-zA-Z]+)in\'', r'\1ing', s)

    s = re.sub(r'([a-zA-Z]+)n\’t', r'\1 not', s)
    # Replacing i'm.
    s = re.sub(r'([iI])\’m', r'\1 am', s)
    # Replacing we've, i've.
    s = re.sub(r'([a-zA-Z]+)\’ve', r'\1 have', s)
    # Replacing i'd, they'd.
    s = re.sub(r'([a-zA-Z]+)\’d', r'\1 had', s)
    # Replacing i'll, they'll.
    s = re.sub(r'([a-zA-Z]+)\’ll', r'\1 will', s)
    # Replacing we're, they're.
    s = re.sub(r'([a-zA-Z]+)\’re', r'\1 are', s)
    # Replacing tryin', doin'.
    s = re.sub(r'([a-zA-Z]+)in\’', r'\1ing', s)

    # Special charecter
    s = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', s)



    # Convert to lowercase
    # Input => s = "ThIs Is A StrInG In UPPERCASE"
    # Output => s = "<this is a string in uppercase"
    s = s.lower()

    # Hashtags
    # Input => s = "#firsthashtag Zalim Dunia #hero9191."
    # Output => s = "<HASHTAG> Zalim Dinia <HASHTAG>."
    s = re.sub(r'#((\w)+[!@#$%^&*()_+-={}[\]:";\'<>,.?/]?)+', '<HASHTAG>', s)

    # Email IDs
    # Input => s = "This is an email address: example@example.com. It also mentions another email: another.example@example.net."
    # Output => s = "This is an email address: <EMAIL>. It also mentions another email: <EMAIL>."
    s = re.sub(
        r'[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.)+(com|in|on|org)(?=[^\w])', '<EMAIL>', s)

    # Mentions
    # Input => s = "This is a tweet mentioning two users: @firstuser and @second_user.""
    # Output => s = "This is a tweet mentioning two users: <MENTION> and <MENTION>."
    # Note : Use it after EmailIDs
    s = re.sub(r'@[a-zA-Z\.\d_-]+', '<MENTION>', s)

    # Numbers:
    # Cases : 1,024
    # Input => s = "The number is 123,456.78 and the percentage is 12.34%. Another number is 987."
    # Output => s = "The number is <NUMBER> and the percentage is <NUMBER>. Another number is <NUMBER>."
    s = re.sub(r'(^|\s)\d+(,(\d+))*(\.\d+)?%?', '<NUMBER> ', s)

    # URLs: a sequence of letters, colons, slashes; a period;
    #       a sequence of letters, slashes, digits;
    #       optionally more periods and sequences;
    #       optionally more slashes and sequences
    # Input => s = "Visit our website at www.example.com/index.html and check out our blog at blog.example.com."
    # Output => s = "Visit our website at <URL> and check out our blog at <URL>."
    s = re.sub(r'(http(s)?:\/\/)?(www\.)?((\w)+\.)+(com|in|on|org|net|co\.in)(\/[\w\-\.]+)*', '<URL>', s)

    # Time expressions of the form HH:MM AM/PM/am/pm
    # Input => s = "The meeting starts at 09:00 AM and ends at 05:00 PM."
    # Output => s = "The meeting starts at <TIME> and ends at <TIME>."
    s = re.sub(r'\s(\d\d:\d\d|\d\d:\d\d:\d\d)\s(AM|PM|am|pm|A.M|P.M|a.m|p.m|A.m|P.m)?', '<TIME>', s)

    # Money expressions in dollars
    # Input => s = "The price of the stock is $100,000 and it has gone up by 5.5% today."
    # Output => s = "The price of the stock is <MONEY> and it has gone up by <MONEY> today."
    s = re.sub(r'(^|\s)\d+(,(\d+))*(\.\d+)?\$', '<MONEY>', s)

    # Repeated punctuation
    # Input => s = "What?? Is this... a joke??? No, it's not!"
    # Output => s = "What? Is this... a joke? No, it's not!"
    s = re.sub(r'([.,?!@#$%^&*()_+-=[\]{}\\\|;\':"<>?])\1+', r'\1', s)


    return s


def tokenize(s):
    # Placing Placeholders and removing repetitive puntuations
    s = clean(s)

    # Clearing Puntuations 
    # s = re.sub(r'[,]\?!;:\-_()"\']', ' ', s)
    # s = re.sub(r'[\,\"\'\_\^\{\}\;\:\(\)\-\]\[\!\@\#\$\%\^\&\*\(\)\_\+\=]', ' ', s)

    # Giving Space besides all puntuations
    # s = re.sub(r'(?<=[^\s])(?=[\.\_\"\'\[\(\)\,\!\?\;])|(?<=[.,!?;])(?=[^\s])', ' ', s)
    s = re.sub(r'([^\w\s])', r' \1 ', s)
    s = re.sub(r'< HASHTAG >', r'<HASHTAG>', s)
    s = re.sub(r'< EMAIL >', r'<EMAIL>', s)
    s = re.sub(r'< URL >', r'<URL>', s)
    s = re.sub(r'< TIME >', r'<TIME>', s)
    s = re.sub(r'< MONEY >', r'<MONEY>', s)
    s = re.sub(r'< NUMBER >', r'<NUMBER>', s)

    # Spacing PlaceHolder
    s=re.sub(r'(<((HASHTAG)|(EMAIL)|(MENTION)|(URL)|(TIME)|(MONEY)|(NUMBER))>)', r' \1 ', s)


    # The following line replaces all strings of more than one space/tab
    # with a single space, and cuts out leading and trailing spaces
    # Input => s = "What   is   this?\tNo,   it's not!"
    # Output => s = "What is this? No, it's not!"
    s = re.sub(r'\s+', ' ', s)
    # Input => s = "   What is this? No, it's not!"
    # Output => s = "What is this? No, it's not!"
    s = re.sub(r'^\s+', '', s)
    # Input => s = "What is this? No, it's not!  "
    # Output => s = "What is this? No, it's not!"
    s = re.sub(r'\s+$', '', s)

    # sentences = nltk.sent_tokenize(s)
    # sentences = re.split(r'[.?]+', s)
    s.split(" ")
    # tokens=s.split()

    return s

    # result = []
    # for sentence in sentences:
    #     if(len(re.findall(r'\b\S+\b', sentence)) > 3):
    #         result.append("<SOS> " + sentence + " <EOS>")  
    # return result



# def write_array_to_file(file_name, array):
#     with open(file_name, 'w') as file:
#         for item in array:
#             file.write(item + '\n')


# def read_file_to_string(file_name):
#     with open(file_name, 'r') as file:
#         return file.read()




# file_name = 'Pride-and-Prejudice-Jane-Austen.txt'
# text = read_file_to_string(file_name)
# # print(text)

# strings = tokenize(text)
# file_name = 'PP_tokenize.txt'
# write_array_to_file(file_name, strings)


# file_name = 'Ulysses-James-Joyce.txt'
# text = read_file_to_string(file_name)
# # print(text)

# strings = tokenize(text)
# file_name = 'U_tokenize.txt'
# write_array_to_file(file_name, strings)

