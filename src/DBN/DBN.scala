package DBN

import breeze.linalg.DenseMatrix
import breeze.numerics.{exp, tanh}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by root on 17-3-20.
  */

case class DBNParamater(momentum: Double,
                        alpha: Double,
                        layer: Int,
                        size: Array[Int],
                        iterate: Int,
                        batchSize: Int) extends Serializable

case class DBNweights(
                      W: DenseMatrix[Double],
                      vW: DenseMatrix[Double],
                      b: DenseMatrix[Double],                      vb: DenseMatrix[Double],
                      c: DenseMatrix[Double],
                      vc: DenseMatrix[Double]) extends Serializable
class DBN(
         private var size: Array[Int],
         private var layer: Int,
         private var momentum: Double,
         private var alpha: Double
         )extends Serializable{

  def this() = this(DBNExt.Architecture, 3, 0.0, 1.0)

  def setSize(size: Array[Int]): this.type ={
    this.size = size
    this
  }

  def setLayer(layer: Int): this.type ={
    this.layer = layer
    this
  }

  def setMomentum(momentum: Double): this.type ={
    this.momentum= momentum
    this
  }

  def setAlpha(alpha: Double): this.type ={
    this.alpha = alpha
    this
  }

  def DBNTrain(train_d: RDD[(DenseMatrix[Double], DenseMatrix[Double])], batchSize: Int, iterate: Int): DBNModel ={
    val sc = train_d.sparkContext

    var dbn_W = DBNExt.InitialW(size)
    var dbn_vW = DBNExt.InitialvW(size)
    var dbn_b = DBNExt.Initialb(size)
    var dbn_vb = DBNExt.Initialvb(size)
    var dbn_c = DBNExt.Initialc(size)
    var dbn_vc = DBNExt.Initialvc(size)

    val weights0 = new DBNweights(dbn_W(0), dbn_vW(0), dbn_b(0), dbn_vb(0), dbn_c(0), dbn_vc(0))
    val paramater = new DBNParamater(momentum, alpha, layer, size, iterate, batchSize)

    val weights1 = RBMtrain(train_d, paramater, weights0)

    dbn_W(0) = weights1.W
    dbn_vW(0) = weights1.vW
    dbn_b(0) = weights1.b
    dbn_vb(0) = weights1.vb
    dbn_c(0) = weights1.c
    dbn_vc(0) = weights1.vc

    var train_d1 = train_d

    for(l <- 2 to paramater.layer - 1){
      val tmp_bcW = sc.broadcast(dbn_W(l - 2))
      val tmp_bcC = sc.broadcast(dbn_c(l - 2))


      val layerHi = train_d1.map(t => {
        val label = t._1
        val xi = DBNExt.sigm(t._2 * tmp_bcW.value.t + tmp_bcC.value.t)
        (label, xi)
      })

      val weighti = new DBNweights(dbn_W(l - 1), dbn_vW(l - 1), dbn_b(l - 1), dbn_vb(l - 1), dbn_c(l - 1), dbn_vc(l - 1))
      val weight2 = RBMtrain(layerHi, paramater, weighti)
      dbn_W(l - 1) = weight2.W
      dbn_vW(l - 1) = weight2.vW
      dbn_b(l - 1) = weight2.b
      dbn_vb(l - 1) = weight2.vb
      dbn_c(l - 1) = weight2.c
      dbn_vc(l - 1) = weight2.vc

      train_d1 = layerHi
    }
    new DBNModel(paramater, dbn_W, dbn_c)
  }

  def RBMtrain(train_t: RDD[(DenseMatrix[Double], DenseMatrix[Double])],
               paramater: DBNParamater,
               weight: DBNweights): DBNweights = {
    val sc = train_t.sparkContext
    var rbm_W = weight.W
    var rbm_vW = weight.vW
    var rbm_b = weight.b
    var rbm_vb = weight.vb
    var rbm_c = weight.c
    var rbm_vc = weight.vc


    val bc_paramater = sc.broadcast(paramater)

    val batchNum = (train_t.count() / bc_paramater.value.batchSize).toInt

    for(i <- 1 to bc_paramater.value.iterate) {
      val splitWeight = Array.fill(batchNum.toInt)(1.0 / batchNum.toInt)
      for (batchIndex <- 1 to batchNum.toInt){
        val bc_rbm_W = sc.broadcast(rbm_W)
        val bc_rbm_vW = sc.broadcast(rbm_vW)
        val bc_rbm_b = sc.broadcast(rbm_b)
        val bc_rbm_vb = sc.broadcast(rbm_vb)
        val bc_rbm_c = sc.broadcast(rbm_c)
        val bc_rbm_vc = sc.broadcast(rbm_vc)


        val train_splits = train_t.randomSplit(splitWeight, System.nanoTime())
        val train_split = train_splits(batchIndex - 1)

        val vhdata = train_split.map(t => {
          val label = t._1

          val v1 = t._2

          /*println(v1.rows + "~" + v1.cols)
          println(bc_rbm_W.value.rows + "~" + bc_rbm_W.value.cols)
          println(bc_rbm_c.value.rows + "~" + bc_rbm_c.value.cols)
          println(v1 * bc_rbm_W.value.t)*/

          val h1 = DBNExt.sigmrnd(v1 * bc_rbm_W.value.t + bc_rbm_c.value.t)
          val v2 = DBNExt.sigmrnd(h1 * bc_rbm_W.value + bc_rbm_b.value.t)
          val h2 = DBNExt.sigm(v2 * bc_rbm_W.value.t + bc_rbm_c.value.t)

          val c1 = h1.t * v1
          val c2 = h2.t * v2

          (label, v1, h1, c2, h2, c1, c2)
        })
        val vw1 = vhdata.map {
          case (lable, v1, h1, v2, h2, c1, c2) =>
            c1 - c2
        }
        val initw = DenseMatrix.zeros[Double](bc_rbm_W.value.rows, bc_rbm_W.value.cols)
        val (vw2, countw2) = vw1.treeAggregate((initw, 0L))(
          seqOp = (c, v) => {
            // c: (m, count), v: (m)
            val m1 = c._1
            val m2 = m1 + v
            (m2, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (m, count)
            val m1 = c1._1
            val m2 = c2._1
            val m3 = m1 + m2
            (m3, c1._2 + c2._2)
          })
        val vw3 = vw2 / countw2.toDouble
        rbm_vW = bc_paramater.value.momentum * bc_rbm_vW.value + bc_paramater.value.alpha * vw3

        val vb1 = vhdata.map {
          case (lable, v1, h1, v2, h2, c1, c2) =>
            (v1 - v2)
        }
        val initb = DenseMatrix.zeros[Double](bc_rbm_vb.value.cols, bc_rbm_vb.value.rows)
        val (vb2, countb2) = vb1.treeAggregate((initb, 0L))(
          seqOp = (c, v) => {
            // c: (m, count), v: (m)
            val m1 = c._1
            val m2 = m1 + v
            (m2, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (m, count)
            val m1 = c1._1
            val m2 = c2._1
            val m3 = m1 + m2
            (m3, c1._2 + c2._2)
          })
        val vb3 = vb2 / countb2.toDouble
        rbm_vb = bc_paramater.value.momentum * bc_rbm_vb.value + bc_paramater.value.alpha * vb3.t

        val vc1 = vhdata.map {
          case (lable, v1, h1, v2, h2, c1, c2) =>
            (h1 - h2)
        }
        val initc = DenseMatrix.zeros[Double](bc_rbm_vc.value.cols, bc_rbm_vc.value.rows)
        val (vc2, countc2) = vc1.treeAggregate((initc, 0L))(
          seqOp = (c, v) => {
            // c: (m, count), v: (m)
            val m1 = c._1
            val m2 = m1 + v
            (m2, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (m, count)
            val m1 = c1._1
            val m2 = c2._1
            val m3 = m1 + m2
            (m3, c1._2 + c2._2)
          })
        val vc3 = vc2 / countc2.toDouble
        rbm_vc = bc_paramater.value.momentum * bc_rbm_vc.value + bc_paramater.value.alpha * vc3.t
        rbm_W = bc_rbm_W.value + rbm_vW
        rbm_b = bc_rbm_b.value + rbm_vb
        rbm_c = bc_rbm_c.value + rbm_vc
      }
    }
    new DBNweights(rbm_W, rbm_vW, rbm_b, rbm_vb, rbm_c, rbm_vc)
  }
}

object DBNExt extends Serializable{
  val Activation_Function = "sigm"
  val Output = "linear"
  val Architecture = Array(10, 5, 1)

  def InitialW(size: Array[Int]): Array[DenseMatrix[Double]] = {

    val n = size.length
    val Weigths = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i), size(i - 1))
      Weigths += d1
    }
    Weigths.toArray
  }

  def InitialvW(size: Array[Int]): Array[DenseMatrix[Double]] = {

    val n = size.length
    val WeigthsV = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i), size(i - 1))
      WeigthsV += d1
    }

    WeigthsV.toArray
  }

  def Initialb(size: Array[Int]): Array[DenseMatrix[Double]] = {

    val n = size.length
    val Bias = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i - 1), 1)
      Bias += d1
    }
    Bias.toArray
  }

  def Initialvb(size: Array[Int]): Array[DenseMatrix[Double]] = {

    val n = size.length
    val BiasV = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i - 1), 1)
      BiasV += d1
    }
    BiasV.toArray
  }

  def Initialc(size: Array[Int]): Array[DenseMatrix[Double]] = {

    val n = size.length
    val CBias = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i), 1)
      CBias += d1
    }
    CBias.toArray
  }

  def Initialvc(size: Array[Int]): Array[DenseMatrix[Double]] = {

    val n = size.length
    val CBiasV = ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = DenseMatrix.zeros[Double](size(i), 1)
      CBiasV += d1
    }
    CBiasV.toArray
  }

  def sigmrnd(P: DenseMatrix[Double]): DenseMatrix[Double] = {
    val s1 = 1.0 / (exp(P * (-1.0)) + 1.0)
    val r1 = DenseMatrix.rand[Double](s1.rows, s1.cols)
    val a1 = s1 :> r1
    val a2 = a1.data.map { f => if (f == true) 1.0 else 0.0 }
    val a3 = new DenseMatrix(s1.rows, s1.cols, a2)

    a3
  }

  def sigmrnd2(P: DenseMatrix[Double]): DenseMatrix[Double] = {
    val s1 = 1.0 / (exp(P * (-1.0)) + 1.0)
    val r1 = DenseMatrix.rand[Double](s1.rows, s1.cols)
    val a3 = s1 + (r1 * 1.0)
    a3
  }

  /**
    * sigm激活函数
    * X = 1./(1+exp(-P));
    */
  def sigm(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val s1 = 1.0 / (exp(matrix * (-1.0)) + 1.0)
    s1
  }

  /**
    * tanh激活函数
    * f=1.7159*tanh(2/3.*A);
    */
  def tanh_opt(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val s1 = tanh(matrix * (2.0 / 3.0)) * 1.7159
    s1
  }

}
