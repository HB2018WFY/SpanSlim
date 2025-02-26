

from flask import Flask, request, jsonify
from entity.Trace import Trace
import asyncio

from others.Sieve.sieve import Sieve

app = Flask(__name__)
sampler = Sieve(tree_num=50, tree_size=128, k=50, threshold=0.3)
count = 0

async def sample(trace):
    global sampler, count
    count += 1
    decision, encode_t, sample_t = sampler.isSample(trace)
    if count % 128 == 0:
        print('before compact: %d' % sampler.getEncodeLength(), end=', ')
        sampler.compact()
        count = 0
        print('after compact: %d' % sampler.getEncodeLength())
    return



@app.route('/collect', methods=['POST'])
def collect():
    data = request.get_json()
    trace = Trace.deserialize(data['trace'])
    # loop = asyncio.new_event_loop()
    # loop.create_task(sample(trace))
    asyncio.run(sample(trace))

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(port=5000)