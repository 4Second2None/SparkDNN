package ANN
import ANN.ANNParameter
import breeze.linalg.DenseMatrix
import org.apache.spark.rdd.RDD

/**
  * Created by root on 17-3-23.
  */
class ANNModel(
                val parameter: ANNParameter,
                val weights: Array[DenseMatrix[Double]]
              )extends Serializable{
  def predict(dataMatrix: RDD[(DenseMatrix[Double], DenseMatrix[Double])]): Array[(DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double])] ={
    val sc = dataMatrix.sparkContext
    val bc_nn_W = sc.broadcast(weights)
    val bc_parameter = sc.broadcast(parameter)

    val train_nnff = ANNExt.ANNff(dataMatrix, bc_parameter, bc_nn_W)

    val predictInfo = train_nnff.map(t => {
      val lable = t.lable
      val error = t.error

      val predictLable = t.nna(bc_parameter.value.layer - 1)

      (lable, predictLable, error)
    })

    weights.foreach(println)
    predictInfo.collect()

  }
}