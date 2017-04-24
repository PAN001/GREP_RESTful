from app import app
from OpenSSL import SSL
from flask_sslify import SSLify

# context = SSL.Context(SSL.SSLv23_METHOD)
# context.use_privatekey_file('/root/OtherServer/panatopos.top_ssl.key')
# context.use_certificate_file('/root/OtherServer/2_panatopos.top.crt')

context = ('ssl_certificates/Self-signed/panatopos.top.crt', 'ssl_certificates/Self-signed/panatopos.top_ssl_without_pass.key')
app.run(debug=True, host='0.0.0.0', port = 443, ssl_context=context)
#app.run(debug=True, host='127.0.0.1', port = 5000, ssl_context=context)
sslify = SSLify(app)
