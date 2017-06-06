package DBN

import breeze.linalg.DenseMatrix

import scala.collection.mutable.ArrayBuffer

/**
  * Created by root on 17-3-20.
  */
class DBNModel(paramater: DBNParamater,
               weigths: Array[DenseMatrix[Double]],
               bias: Array[DenseMatrix[Double]]) {
  def adapt2ANN(yDomain: Int): (Array[Int], Int, Array[DenseMatrix[Double]]) = {
    val size = if(yDomain > 0) {
      val size1 = paramater.size
      val size2 = ArrayBuffer[Int]()
      size2 ++= size1

      size2 += yDomain
      size2.toArray
    }else{
      paramater.size
    }

    val layer = if(yDomain > 0) {
      paramater.layer + 1
    }else{
      paramater.layer
    }

    val WeightsAdBias = ArrayBuffer[DenseMatrix[Double]]()

    for(w <- 0 to weigths.length - 1){
      WeightsAdBias += DenseMatrix.horzcat(bias(w), weigths(w))
    }

    (size, layer, WeightsAdBias.toArray)
  }

  def printInfo(): Unit = {
    println("权重：")
    weigths.foreach(println)
    println("偏置：")
    bias.foreach(println)
  }
}
