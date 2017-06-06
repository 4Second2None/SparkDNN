/**
  * Created by root on 17-3-21.
  */
import java.io.{File, FileInputStream, PrintWriter}

import org.apache.log4j.{Level, Logger}
import DBN.DBN
import ANN.ANN
import CNN.CNN
import breeze.linalg.{DenseMatrix, DenseVector, accumulate, kron, max, min, rot90}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import breeze.numerics._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source


object DeepLearningTest {

  def readMnistFeature(path: String): ArrayBuffer[Seq[Double]]={
    //val d = Source.fromFile(path)

    val in = new FileInputStream(path)

    var b: Array[Byte] = new Array[Byte](16)
    val n = in.read(b)
    val m = (b(0) << 24).toInt + (b(1) << 16).toInt + ((b(2) << 8).toInt) + (b(3).toInt)
    val ni = (b(4) << 24).toInt + (b(5) << 16).toInt + ((b(6) << 8).toInt) + (b(7).toInt)
    val nr = (b(8) << 24).toInt + (b(9) << 16).toInt + ((b(10) << 8).toInt) + (b(11).toInt)
    val nc = (b(12) << 24).toInt + (b(13) << 16).toInt + ((b(14) << 8).toInt) + (b(15).toInt)

    val retbuf = new ArrayBuffer[Seq[Double]]()

    for(j <- 0 to ni - 1){
      val pixelbuf = new ArrayBuffer[Byte]()
      for(i <- 0 to nc * nr - 1){
        pixelbuf += 0
      }
      val pixel = pixelbuf.toArray

      retbuf += pixel.map(Byte.byte2double(_)).toSeq
    }

    retbuf
  }

  def readMnistLable(path: String): Seq[Double]={
    val in = new FileInputStream(path)

    var b: Array[Byte] = new Array[Byte](8)
    val n = in.read(b)
    val m = (b(0) << 24).toInt + (b(1) << 16).toInt + ((b(2) << 8).toInt) + (b(3).toInt)
    val ni = (b(4) << 24).toInt + (b(5) << 16).toInt + ((b(6) << 8).toInt) + (b(7).toInt)


    val retTmp = new ArrayBuffer[Byte]()
    for(i <- 0 to ni - 1){
      retTmp += 0
    }
    val retArray = retTmp.toArray

    retArray.map(Byte.byte2double(_)).toSeq

  }

  def reShapetest(a: Array[Double]): DenseMatrix[Double] = {
    val res = new DenseMatrix(1, a.length, a).reshape(3, 3)
    res
  }
  def expand(a: DenseMatrix[Double], s: Array[Int]): DenseMatrix[Double] = {
    // val a = BDM((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    // val s = Array(3, 2)
    val sa = Array(a.rows, a.cols)
    var tt = new Array[Array[Int]](sa.length)
    for (ii <- sa.length - 1 to 0 by -1) {
      var h = DenseVector.zeros[Int](sa(ii) * s(ii))
      h(0 to sa(ii) * s(ii) - 1 by s(ii)) := 1
      tt(ii) = accumulate(h).data
    }
    var b = DenseMatrix.zeros[Double](tt(0).length, tt(1).length)
    for (j1 <- 0 to b.rows - 1) {
      for (j2 <- 0 to b.cols - 1) {
        b(j1, j2) = a(tt(0)(j1) - 1, tt(1)(j2) - 1)
      }
    }
    b
  }

  def expand1(a: DenseMatrix[Double], s: Array[Int]): DenseMatrix[Double] ={
    val b = DenseMatrix.ones[Double](s(0), s(1))

    val res = kron(a, b)

    res
  }

  def convn(m0: DenseMatrix[Double], k0: DenseMatrix[Double], shape: String): DenseMatrix[Double] = {
    //val m0 = DenseMatrix((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0))
    //val k0 = DenseMatrix((1.0, 1.0), (0.0, 1.0))
    //val m0 = BDM((1.0, 5.0, 9.0), (3.0, 6.0, 12.0), (7.0, 2.0, 11.0))
    //val k0 = BDM((1.0, 2.0, 0.0), (0.0, 5.0, 6.0), (7.0, 0.0, 9.0))
    val out1 = shape match {
      case "valid" =>
        val m1 = m0
        val k1 = rot90(rot90(k0))
        val row1 = m1.rows - k1.rows + 1
        val col1 = m1.cols - k1.cols + 1
        var m2 = DenseMatrix.zeros[Double](row1, col1)
        for (i <- 0 to row1 - 1) {
          for (j <- 0 to col1 - 1) {
            val r1 = i
            val r2 = r1 + k1.rows - 1
            val c1 = j
            val c2 = c1 + k1.cols - 1
            val mi = m1(r1 to r2, c1 to c2)
            m2(i, j) = (mi :* k1).sum
          }
        }
        m2
      case "full" =>
        var m1 = DenseMatrix.zeros[Double](m0.rows + 2 * (k0.rows - 1), m0.cols + 2 * (k0.cols - 1))
        for (i <- 0 to m0.rows - 1) {
          for (j <- 0 to m0.cols - 1) {
            m1((k0.rows - 1) + i, (k0.cols - 1) + j) = m0(i, j)
          }
        }
        val k1 = rot90(rot90(k0))
        val row1 = m1.rows - k1.rows + 1
        val col1 = m1.cols - k1.cols + 1
        var m2 = DenseMatrix.zeros[Double](row1, col1)
        for (i <- 0 to row1 - 1) {
          for (j <- 0 to col1 - 1) {
            val r1 = i
            val r2 = r1 + k1.rows - 1
            val c1 = j
            val c2 = c1 + k1.cols - 1
            val mi = m1(r1 to r2, c1 to c2)
            m2(i, j) = (mi :* k1).sum
          }
        }
        m2
    }
    out1
  }
  def matrixTest(): Unit ={
    val k = Array[Double](1.0, 0.5776179005536245,0.19581350473634737,0.14698070783474912,0.9106297522517818,0.22547544271272746)
    val km = new DenseMatrix[Double](1, k.length, k)
    val ws = loadW("/opt/ANNWeight")

    val res = tanh((km * ws(0).t) * (2.0 / 3.0)) * 1.7159
    println(res)
  }

  def CNNTest(sc: SparkContext, d: Array[(DenseMatrix[Double], DenseMatrix[Double])]): Unit = {
    val train_data = sc.parallelize(d, 10)
    val train_data1 = train_data.map(f => (f._1, f._2))

    val CNNmodel = new CNN().setInputSize(new DenseMatrix[Double](1, 2, Array(28.0, 28.0))).
      setArchitecture(Array("i", "c", "s", "c", "s")).
      setLayer(5).
      setOutDimension(10).
      setOutSizeList(Array(0.0, 6.0, 0.0, 12.0, 0.0)).
      setKernelSizeList(Array(0.0, 5.0, 0.0, 5.0, 0.0)).
      setScales(Array(0.0, 0.0, 2.0, 0.0, 2.0)).
      setAlpha(1.0).train(train_data1, 1000, 10)

  }

  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("NetTest").setMaster("local[4]").set("spark.executor.memory", "8g")
    val sc = new SparkContext(conf)
    //DBNTest(sc)
    //ANNTest(sc)
    //val a = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    //println(reShapetest(a))
    /*val m0 = DenseMatrix((1.0, 1.0, 1.0, 1.0, 1.0, 1.5), (0.0, 0.0, 1.0, 1.0, 1.0, 1.5), (0.0, 1.0, 1.0, 0.0, 1.0, 1.5), (0.0, 1.0, 1.0, 0.0, 1.0, 1.5), (0.0, 0.0, 1.0, 1.0, 1.0, 1.5), (0.0, 1.0, 1.0, 0.0, 1.0, 1.5))
    val k0 = DenseMatrix((1.0, 1.0), (0.0, 1.0))
    val tmp = convn(m0, k0, "valid")

    println(tmp)
    println("****************")
    val tmp1 = tmp(::, 0 to -1 by 2).t + 0.0
    println(tmp1)
    println("****************")
    val tmp2 = tmp1(::, 0 to -1 by 2).t + 0.0
    println(tmp2)*/

    val features = readMnistFeature("/root/IdeaProjects/TensorFlowTest/MNIST_data/t10k-images.idx3-ubyte").toSeq


    val labels = readMnistLable("/root/IdeaProjects/TensorFlowTest/MNIST_data/t10k-labels.idx1-ubyte")

    val lablefeature = labels.map(d => {
      val s = Array[Double](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      s(d.toInt) = 1.0
      s
    })

    val MatrixArray = new ArrayBuffer[(DenseMatrix[Double], DenseMatrix[Double])]()
    for(i <- 0 to lablefeature.length - 1){
      MatrixArray += ((new DenseMatrix(1, lablefeature(i).length, lablefeature(i)), new DenseMatrix(1, features(i).length, features(i).toArray).reshape(28, 28)))
    }
    CNNTest(sc, MatrixArray.toArray)

    //val kk: Seq[Int] = Seq(1,2,3,4,5,6)

    //println(kk.slice(1, 4))
  }

  def loadW(path: String): Array[DenseMatrix[Double]] = {
    val Ws = ArrayBuffer[DenseMatrix[Double]]()


    var darray = Array[Double]()
    var row = 0
    var col = 0
    for(line<-Source.fromFile(path).getLines){
      if(line != "*********"){
        val sa = line.split(",").map(_.toDouble)
        col = sa.length
        darray = darray ++ sa
        row = row + 1
      }else{
        val dm = new DenseMatrix[Double](col, row, darray).t
        darray = Array[Double]()
        row = 0
        col = 0
        Ws += dm
      }
    }
    Ws.toArray
  }

  def outW(size: Array[Int]): Unit ={
    val writer = new PrintWriter(new File("/opt/ANNWeight"))
    val WArray = ArrayBuffer[DenseMatrix[Double]]()

    for(i <- 1 to size.length - 1){
      val dm1 = DenseMatrix.rand(size(i), size(i - 1) + 1)

      dm1 :-= 0.5

      val rate = 2 * 4 * sqrt(6.0 / (size(i) + size(i - 1)))

      val dm2 = dm1 :* rate
      val dm = dm2.asInstanceOf[DenseMatrix[Double]]

      for(r <- 0 to dm.rows - 1){
        for(c <- 0 to dm.cols - 1){
          writer.print(dm.valueAt(r, c))
          if(c == dm.cols - 1)
            writer.println()
          else
            writer.print(",")
        }
      }
      writer.println("*********")
    }
    writer.close()
  }

  def loadTestData(sc: SparkContext, path: String): Array[DenseMatrix[Double]] = {
    val data = sc.textFile(path).map(d => {
      val splits = d.split(":")

      val label = splits(0).toDouble
      val features = splits(1).split(",").map(_.toDouble)

      val mil = new DenseMatrix(1, 1, Array[Double](label))
      val mif = new DenseMatrix(1, features.length, features)

      DenseMatrix.horzcat(mil, mif)
    }).collect()
    data
  }

  def saveTestData(path: String): Unit ={
    val writer = new PrintWriter(new File(path))
    val sample_n1 = 100
    val sample_n2 = 5
    val randsamp1 = RandSampleData.RandM(sample_n1, sample_n2, -10, 10, "sphere")
    // 归一化[0 1]
    val normmax = max(randsamp1(::, breeze.linalg.*))

    val normmin = min(randsamp1(::, breeze.linalg.*))

    val norm1 = randsamp1 - (DenseMatrix.ones[Double](randsamp1.rows, 1)) * normmin
    val norm2 = norm1 :/ ((DenseMatrix.ones[Double](norm1.rows, 1)) * (normmax - normmin))


    // 转换样本train_d
    for (i <- 0 to sample_n1 - 1) {
      val mi = norm2(i, ::)
      val mi1 = mi.inner
      val mi2 = mi1.toArray
      writer.print(mi2(0) + ":")
      for(i <- 1 to mi2.length - 1){
        writer.print(mi2(i))
        if(i != mi2.length - 1)
          writer.print(",")
        else
          writer.println()
      }
    }
    writer.close()
  }

  def ANNTest(sc: SparkContext): Unit = {
    val sample_n1 = 100
    val sample_n2 = 5
    val randsamp1 = RandSampleData.RandM(sample_n1, sample_n2, -10, 10, "sphere")
    // 归一化[0 1]
    val normmax = max(randsamp1(::, breeze.linalg.*))

    val normmin = min(randsamp1(::, breeze.linalg.*))

    val norm1 = randsamp1 - (DenseMatrix.ones[Double](randsamp1.rows, 1)) * normmin
    val norm2 = norm1 :/ ((DenseMatrix.ones[Double](norm1.rows, 1)) * (normmax - normmin))


    // 转换样本train_d
    val randsamp2 =ArrayBuffer[DenseMatrix[Double]]()
    for (i <- 0 to sample_n1 - 1) {
      val mi = norm2(i, ::)
      val mi1 = mi.inner
      val mi2 = mi1.toArray
      val mi3 = new DenseMatrix(1, mi2.length, mi2)

      randsamp2 += mi3
    }

    //val data = loadTestData(sc, "/opt/ANNData")
    val randsamp3 = sc.parallelize(randsamp2, 10)
    //sc.setCheckpointDir("/home/cwl/gtest")
    //randsamp3.checkpoint()
    //val ann = new ANN(Array(3, 6, 4, 3, 4, 2, 1), 5, 0.5, "sigmoid", 100)
    val ann = new ANN(Array(5, 7, 6, 2, 1), 5, "tan", "sigm", 0.8, 0.3, Array(DenseMatrix.zeros[Double](1, 1)))
    //ann.setInitW(loadW("/opt/ANNWeight"))

    val train_d = randsamp3.map(f => (new DenseMatrix(1, 1, f(::, 0).data), f(::, 1 to -1)))

    //train_d.collect()

    val model = ann.train(train_d, 20, 20)

    val predictdata = model.predict(train_d)

    predictdata.foreach(println)
  }
  def DBNTest(sc: SparkContext): Unit ={
    Logger.getRootLogger.setLevel(Level.WARN)
    val data_path = "/root/IdeaProjects/SparkMLlibDeepLearn/src/Data/data1.txt"
    val examples = sc.textFile(data_path).cache()
    val train_d1 = examples.map { line =>
      val f1 = line.split("\t")
      val f = f1.map(f => f.toDouble)
      //val id = f(0)
      val y = Array(f(0))
      val x = f.slice(1, f.length)
      (1, new DenseMatrix(1, y.length, y), new DenseMatrix(1, x.length, x))
    }
    val train_d = train_d1.map(f => (f._2, f._3))
    val opts = Array(100.0, 20.0, 0.0)

    //3 设置训练参数，建立DBN模型
    val dbnModel = new DBN().
      setSize(Array(5, 7)).
      setLayer(2).
      setMomentum(0.1).
      setAlpha(1.0).DBNTrain(train_d, 100, 20)

    //dbnModel.printInfo()

    val annParam = dbnModel.adapt2ANN(1)

    val annSize = annParam._1
    val annLayer = annParam._2
    val annWeigths = annParam._3

    /*annSize.foreach(println)
    println("***")
    println(annLayer)
    println("***")
    annWeigths.foreach(println)*/

    val annModel = new ANN().
      setSize(annSize).
      setActivation_function("sigm").
      setLastout_function("sigm").
      setLayer(annLayer).
      setInitW(annWeigths).
      train(train_d, 100, 50)

    val predictdata = annModel.predict(train_d)

    predictdata.foreach(println)
  }
}
