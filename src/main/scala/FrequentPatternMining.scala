import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.fpm.FPGrowth


object FrequentPatternMining {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    if (args.length != 3) {
      println("I/p and O/p filepath needed")
    }
    Logger.getLogger("labAssignment").setLevel(Level.OFF)
    spark.sparkContext.setLogLevel("ERROR")
    val orderProductsData = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(args(0))
    val productsData = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(args(1))
    val filteredRDD = orderProductsData.select("order_id","product_id").rdd
      .map(row => (row(0).toString(), row(1).toString().toInt))
    val groupedData = filteredRDD.groupByKey()
    val transactions = groupedData.mapValues(_.toArray).collectAsMap().values.toArray
    val transactionsRDD = spark.sparkContext.parallelize(transactions)
    val fpg = new FPGrowth()
      .setMinSupport(0.01)
      .setNumPartitions(100)
    val model = fpg.run(transactionsRDD)
    //top 10 pattern
    val freqPattern = model.freqItemsets.map ( itemset =>
      (itemset.items,itemset.freq)
    ).sortBy(-_._2).take(10)

    val printContent = new StringBuilder()
    freqPattern.foreach { itemset =>
      itemset._1.foreach(
        prod => printContent.append(productsData.filter(productsData("product_id")===prod).select("product_name").take(1)(0).toString())
      )
      printContent.append(", " + itemset._2+"\n")
    }



    //top 10 rules
    val minConfidence = 0.1
    val associationRules = model.generateAssociationRules(minConfidence).map ( rule =>
      ((rule.antecedent,rule.consequent),rule.confidence)
    ).collect().sortBy(-_._2).take(10)

    val associationRulesArray = associationRules.map(rule => ((rule._1._1.map(ante => productsData.filter(productsData("product_id")===ante).select("product_name").take(1)(0).toString()),
      rule._1._2.map(cons => productsData.filter(productsData("product_id")===cons).select("product_name").take(1)(0).toString())),rule._2)
    )

    associationRulesArray.foreach{ rule =>
      printContent.append("\n")
      rule._1._1.foreach{ item =>
        printContent.append(item)

      }
      printContent.append(" => ")
      rule._1._2.foreach{
        printContent.append(_)
      }
      printContent.append(": ")
      printContent.append(rule._2)
    }

    val printRdd = spark.sparkContext.parallelize(Seq(printContent))
    printRdd.saveAsTextFile(args(2))



  }
}
