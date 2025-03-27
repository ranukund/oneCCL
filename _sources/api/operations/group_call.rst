.. _group_call:

Group Calls
************

Group calls can merge multiple calls into a single group operation. These operations are initiated and finalized using primary functions: ``group_start()`` and ``group_end()``.  

``Group_start``
===============

.. code:: cpp

    void group_start() 

Start a group call. You can use the group_start() function to initiate a group call operation which indicates that successive operations should not get blocked due to CPU synchronization.  

.. code:: cpp

    void group_end() 

End a group call. The ``group_end()`` call returns when all the operations between ``group_start()`` and ``group_end()`` have been enqueued for execution, but not necessarily completed.  

 
.. note:: Currently, group calls are only supported for point-to-point operations.  

