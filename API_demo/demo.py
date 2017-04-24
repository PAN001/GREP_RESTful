import httplib

url = ""

conn = httplib.HTTPConnection("")
conn.request(method="GET",url=url) 

response = conn.getresponse()
res= response.read()
print res