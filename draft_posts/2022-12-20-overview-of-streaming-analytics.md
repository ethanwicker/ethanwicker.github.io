---
layout: post
title: "Streaming Analytics"
subtitle: "An Overview"
comments: false
---

Streaming Analytics refers to the processing and analyzing of data records continuously instead of in regular batches. Streams are triggered by specific events as the result of an action or set of actions. Examples of these triggering events might include financial transactions, thermostat readings, website purchases, or phone calls. Streaming Analytics is also known as event stream processing.
In the Streaming Analytics paradigm, it can be helpful to think of data being streamed through a query. This is in contrast to the batch processing paradigm, where a query can be thought of as being submitted over an entire table or dataset. This is an embellishment and oversimplification, but helpful to illustrate the paradigm switch between streaming and batch processing.
As an example, imagine we’re interested in sending consumers an alert anytime their indoor home temperature is below 55 degrees. In this example, data is collected from thermostats every hour and streamed through a SQL query that returns TRUE anytime temp is less than or equal to 55 degrees. If a TRUE is returned, a downstream process is kicked off to send a mobile alert to the home owner. In this streaming example, the query can be thought of as always being active and waiting for new data to be sent through it. In contrast, a batch application might store these hourly thermostat readings in a database, and a scheduled query might run over all the new readings every six hours.

TOC HERE?

### Stream Processing Tools

For Streaming Analytics, a stream processing tool is needed to manage the movement of the streaming data. Examples of these include Apache Kafka, Amazon Kinesis, and Google Cloud Dataflow. Apache Kafka might be the most common of these. Apache Kafka is almost always configured to process (move) data in parallel threads. Apache Kafka is typically configured to guarantee data that will be delivered at least once or exactly once, but it does guarantee how that data will be sent. It might be sent over one thread or multiple threads. Because of this, data can be received late, out-or-order, or (sometimes) duplicated. This point is crucial in the context of streaming analytics and stream processing engines.

### Stream Processing Engines

In addition to stream processing tools like Apache Kafka, a stream processing engine is needed to facilitate queries on top of streamed data. Examples of these include Apache Spark Structured Streaming (formerly Spark Streaming), Apache Flink (which is also a stream processing tool), Apache Storm, Spark SQL, and RapidMind. Because data can be received late, out-of-order, or duplicated, the streaming processing engine must be able to handle and reason about all of this.

### The Three V’s of Streaming Analytics

Occasionally discussed, the three V’s of Streaming Analytics refer to:
- Volume - refers to the (sometimes extremely large) amount of data being stored by many modern organizations
- Velocity - refers to the amount of data being streamed
- Variety - refers to the variety of data and formats being streamed (e.g., JSON, binary, raw text, images, etc.)

### Spark Structured Streaming

![2022-12-20-oveview-of-streaming-analytics-fig-1.png](/assets/img/2022-12-20-oveview-of-streaming-analytics-fig-1.png){: .mx-auto.d-block :}

Spark Structured Streaming (formerly Spark Streaming) is a stream processing engine built on the Spark SQL engine. Structured Streaming is scalable, fault-tolerant, supports event-time windows and stream-to-batch joins, and guarantees exactly-once stream processing. Internally, Structured Streaming processes queries using a micro-batch processing engine which can achieve end-to-end latencies as low as 100 milliseconds.

FOR MORE DETAILS SEE https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html

#### Fault Tolerance

In the event Structured Streaming fails (i.e., a fault occurs), end-to-end exactly-once guarantees are ensured. Upon restarting, Structured Streaming will not duplicate or miss any processing. This is achieved via checkpoint and write-ahead logging. This is in contrast to some engines that ensure at-least-once guarantees under a failure, in which data will not be missed but may be duplicated.
When processing streaming data, Structured Streaming will generate a Result Table. Whenever this Result Table is updated, we might to write it to external storage. Structured Streaming provides three output modes:
- Complete Mode - the entire Result Table will be written to external storage
- Append Mode - only the new rows appended in the Result Table since the last trigger will be written to the external storage 
- Update Mode - only the rows that were updated in the Result Table since the last trigger will be written to the external storage

#### Window Operations on Event Time

Event time refers to the actual time the event occurred. This is in contrast to the actual time Spark Structured Streaming receives and
processes the event, commonly referred to as processing time.

Aggregations over a sliding event time window are conceptually very similar to grouped aggregations, and are referred to as window operations.

For example, we may be interested in counting the number of detected furnace ignitor failures every 10 minutes. The tumbling window here would be 10 minutes.

#### Handling Late Data and Watermarking

Late-arriving data can be handled via watermarking. Watermarks tell Structure Streaming how long to wait for late-arriving data, and require
a *threshold* value.

For example, if we are counting the number of detected furnace ignitor failures via a 10 minute sliding window, we could define a watermark threshold to be 5 minutes. If an event that occurred at 12:08 was received at 12:13, the event would be correctly included in the count of furnace ignitor failures for the 12:00-12:10 interval. However, if the event arrived at 12:17 - which is past our 5 minute watermark threshold - the event would be dropped entirely and would not be included in our count at all.

#### Types of Time Windows

Spark Structured Streaming supports three types of time windows:

- Tumbling (fixed) Windows - a series of fixed-sized, non-overlapping and contiguous time intervals. An input can only be bound to a single window
- Sliding Windows - similar to the tumbling windows from the point of being “fixed-sized”, but windows can overlap if the duration of slide is smaller than the duration of window, and in this case an input can be bound to the multiple windows
- Session Windows - have a dynamic size of the window length, depending on the inputs. A session window starts with an input, and expands itself if following input has been received within gap duration. For static gap duration, a session window closes  when there’s no input received within gap duration after receiving the latest input

| ![2022-12-20-oveview-of-streaming-analytics-fig-2.png](/assets/img/2022-12-20-oveview-of-streaming-analytics-fig-2.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** [Apache Spark structured streaming programming guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) |

#### Join Operations

Structured Streaming supports a variety of join operations, including streaming-to-static data joins and streaming-to-streaming data joins.

### Considerations

In general, streaming applications can require more planning and forethought than batch processing applications. For example, thought must be given to how to handle delayed data, and what amount of incoming data latency is acceptable.

Below are some considerations for building and managing streaming applications:
- Does the business problem actually require some action or information within a few seconds or few minutes? Does the current batch processing infrastructure support this need and is a streaming approach actually required? Could a batch processing job that runs every x minutes be a simpler approach for your problem?
- For different applications and problems, what degree of data latency is acceptable? Seconds? Minutes? If the streaming processing engine is slowed or disrupted, how detrimental will this be?
- Who will manage the streaming application when deployed in production? Will the management of the application be reasonable? How will your application handle delayed data? After what length of time is it appropriate to ignore delayed data?
- Does your application need both streaming data as well as statically stored data? Is the overall processing time to fetch data from multiple sources - and possibly perform joins - reasonable for your needs?
- Will your application be robust enough to handle differences and nuances in the incoming data stream?
- Does an existing data stream work for your needs? Does a new stream actually need to be created and managed?
- What monitoring is needed to monitor your data stream and overall application?
