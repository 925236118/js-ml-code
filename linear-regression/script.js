import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { input } from '@tensorflow/tfjs'


window.onload = () => {
  const xs = [1, 2, 3, 4]
  const ys = [1, 3, 5, 7]

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
  model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) })

  const inputs = tf.tensor(xs)
  const labels = tf.tensor(ys)

  model.fit(inputs, labels, {
    batchSize: 4,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  })
    .then(() => {
      // const output = model.predict(tf.tensor([5]))
      // output.print()
      // console.log(output.dataSync()[0])


      let datas = Array.from(Array(30), () => Math.random() * 5)
      tfvis.render.scatterplot(
        { name: '样本及预测结果' },
        {
          series: ['样本', '预测'],
          values: [
            xs.map((x, i) => ({ x, y: ys[i] })),
            datas.map((x, i) => {
              return {
                x: x,
                y: model.predict(tf.tensor([x])).dataSync()[0]
              }
            })
          ]
        },
        { xAxisDomain: [0, 6], yAxisDomain: [0, 10] }
      )
    })
}