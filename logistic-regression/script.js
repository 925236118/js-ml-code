import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data'

window.onload = () => {
  const data = getData(400)

  const model = tf.sequential()
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [2],
    activation: 'sigmoid'
  }))
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
  })

  const inputs = tf.tensor(data.map(p => [p.x, p.y]))
  const labels = tf.tensor(data.map(p => p.label))
  model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  })
    .then(() => {
      let datas = Array.from(Array(400), () => ([Math.random() * 8 - 4, Math.random() * 8 - 4]))
      tfvis.render.scatterplot(
        { name: '逻辑回归训练数据以及预测数据' },
        {
          series: ['1', '0', '预测'],
          values: [
            data.filter(p => p.label === 1),
            data.filter(p => p.label === 0),
            datas.map(d => ({ x: d[0], y: d[1], label: model.predict(tf.tensor([[...d]])).dataSync()[0] }))
          ]
        }
      )
    })

  // const output = model.predict(tf.tensor([[2, 1]]))
  // console.log(output.dataSync()[0])
}
