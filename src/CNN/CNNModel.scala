package CNN

import breeze.linalg.DenseMatrix
import org.apache.spark.rdd.RDD

/**
  * Created by root on 17-5-2.
  */
class CNNModel(layers: Array[CNNLayers], finalWeight: DenseMatrix[Double], finalBias: DenseMatrix[Double]) extends Serializable{

  def predict(data: RDD[(DenseMatrix[Double], DenseMatrix[Double])]): Unit ={
    val sc = data.sparkContext
    val Wbc = sc.broadcast(finalWeight)
    val Bbc = sc.broadcast(finalBias)
    val Layersbc = sc.broadcast(layers)

    val predictRes = CNN.CNNExt.CNNff(data, Layersbc, Bbc, Wbc)

    predictRes.map(v => {
      val rawlable = v._1
      val predictlable = v._4

      (rawlable, predictlable)
    })
  }

  def accrate(res: RDD[(DenseMatrix[Double], DenseMatrix[Double])]): Double ={
    val r = res.map(v => {
      val a1 = v._1.toArray
      val a2 = v._2.toArray

      var t1 = Double.MinValue
      var n1 = 0
      var t2 = Double.MinValue
      var n2 = 0

      for(i <- 0 to a1.length - 1){
        if(a1(i) > t1){
          t1 = a1(i)
          n1 = i
        }
      }

      for(i <- 0 to a2.length - 1){
        if(a2(i) > t2){
          t2 = a2(i)
          n2 = i
        }
      }
      if(n1 == n2){
        (1, 1)
      }else{
        (0, 1)
      }
    }).reduce((d1, d2) => {
      (d1._1 + d2._1, d1._2 + d2._2)
    })

    r._1.toDouble / r._2.toDouble
  }

}
